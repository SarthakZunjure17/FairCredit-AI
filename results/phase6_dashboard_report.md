# Phase 6 Final Engineering Report: FairCredit-AI Research Dashboard Implementation

**Project Title:** FairCredit-AI: A Responsible AI Credit Model Auditing Framework  
**Tagline:** Evaluating Fairness, Performance, and Explanation Consistency in Credit Scoring Models  
**Institution:** Department of Computer Science and AI, Vishwakarma Institute of Technology, Pune  
**Date:** October 7, 2026  
**Auditing Status:** COMPLETE & VERIFIED  

---

## 1. Dashboard Architecture

The FairCredit-AI Research Dashboard is constructed as an analytical, decision-support Streamlit application tailored for model-risk managers, fair lending officers, and responsible-AI researchers.

### Architecture Overview
```text
dashboard/
├── app.py                              # Master Streamlit application entrypoint
├── data_loader.py                      # Programmatic caching data loader (st.cache_data)
├── components.py                       # Modular UI components, metric cards, Altair & Matplotlib charts
└── data/
    └── individual_feature_shap.parquet # Frozen individual feature attributions (167,700 rows)
```

### Architectural Design Decisions
1. **Zero Runtime Recomputation:** The dashboard strictly serves as an auditing and visualization layer over frozen research artifacts. Models are never retrained on page load, splits are never recalculated, and no random seeds are mutated.
2. **Programmatic Data Ingestion:** All metrics, confidence intervals, and subgroup statistics are ingested programmatically via `dashboard/data_loader.py`. Numerical values are never hardcoded into UI templates.
3. **Strict N/A Preservation:** For decision-level post-processing (`ThresholdOptimizer`) and randomized reductions (`ExponentiatedGradient`), continuous ROC-AUC metrics are strictly preserved and visibly displayed as `"N/A — continuous ranking score not available under the locked evaluation protocol"`. Under no circumstances are missing ranking scores replaced with zero.
4. **Instant Individual Explanations (<50ms):** Feature-level baseline vs. reweighed attributions across all $N=2,100$ evaluated applicant cases were pre-extracted from the locked Phase 4 models and stored in a compressed columnar file (`individual_feature_shap.parquet`). Selection of any borrower is instantaneous without model re-fitting.
5. **Restrained Model-Risk Visual Design:** Developed with an accessible, high-contrast financial palette (Slate `#1E293B`, Royal Blue `#2563EB`, Teal `#0D9488`, Rose `#E11D48`, Amber `#D97706`) avoiding gamification, unnecessary animations, or decorative AI imagery.

---

## 2. Pages / Tabs Implemented

The application is structured into seven distinct auditing tabs:

| Tab | Section Title | Primary Functionality & Features |
| :---: | :--- | :--- |
| **1** | **Overview** | Outlines framework scope, 2 benchmark datasets, 3 model architectures, 3 mitigation levels, and experimental focus on Baseline vs. Reweighing. |
| **2** | **Model Audit** | Interactive drop-down selectors (`Dataset`, `Model`, `Mitigation Method`) with before/after metric cards for Accuracy, Balanced Accuracy, Precision, Recall, F1, and ROC-AUC (with explicit `N/A` notices). |
| **3** | **Fairness Analysis** | Multi-dimensional group fairness audit (EOD, EOpp, DPD, Approval DI, Distance from Parity). Includes Altair before/after bar charts and a 1.0 parity reference line. Explicitly highlights that fairness criteria can diverge. |
| **4** | **Explanation Consistency** | Cohort-level stability analysis for Baseline vs. Reweighing. Displays Mean Cosine Similarity (with 95% bootstrap CIs), Top-5 Overlap, Spearman Rank correlation, sample size, and cross-model architectural comparisons. |
| **5** | **Individual Explanation** | Borrower-level audit for $N=2,100$ test applicants. Selectable applicant ID, individual metrics, top positive/negative decision drivers, horizontal bar chart, and explicit prediction space labeling (Log-odds margin vs. Class-1 probability). |
| **6** | **Research Findings** | Interactive presentation of frozen Phase 5 findings, the central empirical conclusion, case studies (German RF, Taiwan RF, Taiwan XGBoost), and the full 6-row multi-objective research matrix with sortable headers. |
| **7** | **Methodology & Limitations** | In-depth documentation of the three intervention levels, justifications for excluding post-processing and reductions from scoring SHAP, 11 stated research limitations, and prominent regulatory disclaimers. |

---

## 3. Data Sources

All dashboard views draw directly from the frozen authoritative research artifacts located in `results/`:

1. `results/mitigation_master_results.csv`: Complete 18-experiment table covering all performance and group fairness metrics across Baseline and Mitigated pipelines.
2. `results/final_research_results.csv`: Master frozen table containing the 6 primary Baseline vs. Reweighing rows with exact deltas, relative percentage changes, 95% bootstrap CIs, and consistency levels.
3. `results/shap_summary.csv`: Summary distribution metrics (mean, median, std, 95% CIs) for Cosine Similarity, Top-5 Overlap, Spearman correlation, and nearest-neighbor comparisons.
4. `results/shap_group_metrics.csv`: Subgroup-stratified explanation consistency metrics partitioned by protected demographic status (`sex`).
5. `results/shap_individual_metrics.csv`: Individual consistency records for all $N=2,100$ applicant explanations.
6. `results/fairness_performance_explanation_matrix.csv`: Categorical multi-objective synthesis profiles for the 6 primary configurations.
7. `dashboard/data/individual_feature_shap.parquet`: Pre-extracted feature-level SHAP attributions for all evaluated applicant explanations (167,700 rows $\times$ 7 columns).

---

## 4. Validation Results

The automated Phase 6 verification test suite (`scripts/verify_dashboard_data.py`) was executed to confirm data integrity across 10 mandatory checks:

```text
================================================================================
RUNNING PHASE 6 DASHBOARD DATA VERIFICATION SUITE
================================================================================

[Check 1/10] Verifying existence of all required source files...
  [OK] Found: results\mitigation_master_results.csv
  [OK] Found: results\phase4_explanation_results.csv
  [OK] Found: results\shap_summary.csv
  [OK] Found: results\shap_group_metrics.csv
  [OK] Found: results\shap_individual_metrics.csv
  [OK] Found: results\final_research_results.csv
  [OK] Found: results\fairness_performance_explanation_matrix.csv
  [OK] Found: dashboard\data\individual_feature_shap.parquet
  PASSED: All expected files exist.

[Check 2/10] Verifying the six primary Reweighing research rows...
  PASSED: All 6 primary Reweighing rows present: [('german', 'logistic_regression'), ('german', 'random_forest'), ('german', 'xgboost'), ('taiwan', 'logistic_regression'), ('taiwan', 'random_forest'), ('taiwan', 'xgboost')]

[Check 3/10] Verifying Phase 3 metrics match between master and final results...
  PASSED: Phase 3 authoritative metrics verified across all 6 configurations.

[Check 4/10] Verifying Phase 4 explanation consistency metrics match authoritative values...
  PASSED: Phase 4 authoritative explanation consistency metrics verified.

[Check 5/10] Verifying N/A ROC-AUC values are preserved and never converted to zero...
  PASSED: All 12 ExponentiatedGradient & ThresholdOptimizer mitigated_roc_auc values are strictly 'N/A' (not zero).

[Check 6/10] Verifying Taiwan/German datasets are correctly identified across tables...
  PASSED: Both 'taiwan' and 'german' properly identified in all tables.

[Check 7/10] Verifying all three model architectures are present...
  PASSED: All 3 model architectures present in all tables.

[Check 8/10] Verifying individual SHAP records have valid applicant IDs...
  PASSED: 2,100 applicant records verified across German (N=200) and Taiwan (N=500).

[Check 9/10] Verifying zero leakage of protected demographic attributes into model features...
  PASSED: Feature space confirmed 100% clean of protected sex attributes.

[Check 10/10] Verifying dashboard data loader functions execute cleanly...
  PASSED: All dashboard tables loaded successfully without schema or parsing errors.

================================================================================
ALL 10 PHASE 6 DASHBOARD VERIFICATION CHECKS PASSED SUCCESSFULLY.
================================================================================
```

In addition, headless interactive testing via `streamlit.testing.v1.AppTest` verified that:
- The initial layout renders without exception across all 7 tabs.
- Dataset, model, and mitigation selectboxes trigger clean state updates.
- ThresholdOptimizer selection correctly preserves and formats the `N/A` ROC-AUC notice.
- Selecting individual applicants dynamically loads and renders the horizontal bar attribution chart without errors.

---

## 5. Research Traceability

To maintain strict traceability for bank model-risk audits:
- Every table and chart displays a visible research source indicator (e.g., `📁 Research Data Source: results/mitigation_master_results.csv`).
- Individual explanations clearly state the output space being explained (e.g., margin/log-odds for Logistic Regression and XGBoost, class-1 probability for Random Forest).
- Users are explicitly warned that cross-architecture attribution comparisons are invalid due to disparate prediction spaces.
- The UI maintains a prominent disclaimer banner on all pages stating:  
  *“Research Prototype — Model Auditing & Responsible AI Decision Support — Not for Automated Credit Decisions.”*

---

## 6. Stated Research Limitations

The dashboard explicitly communicates the 11 methodological boundaries established in Phase 5:
1. **Benchmark Datasets:** Limited to two canonical public datasets (Taiwan Credit Card Default and Statlog German Credit).
2. **Fixed Train/Test Split:** Evaluated on deterministic 80/20 stratified splits (`random_state=42`).
3. **Cohort Scope:** Explanation consistency evaluated on $N=500$ applicants for Taiwan and $N=200$ applicants for German.
4. **German Sample Size:** German dataset is relatively small ($N=1,000$ total), creating wider confidence intervals.
5. **Mitigation Scope for SHAP:** Primary explanation-consistency experiment focuses on Reweighing.
6. **ThresholdOptimizer Scope:** Operating at decision threshold level post-hoc, it leaves the baseline scoring function unchanged.
7. **ExponentiatedGradient Scope:** Solved as an ensemble reduction, it lacks a continuous scoring function under the evaluation protocol.
8. **Sample of Configurations:** Six model-dataset pairs form the basis of cross-objective synthesis.
9. **Exploratory Association:** Observed relationships between fairness metrics and consistency cannot establish causality.
10. **Generalizability:** Findings reflect audited configurations and may not generalize to all production portfolios or alternative credit features.
11. **Regulatory Scope:** Prototype does not replace human underwriting or compliance review.

---

## 7. Run Instructions

### Virtual Environment Setup
Ensure dependencies from `requirements.txt` are installed:
```powershell
pip install -r requirements.txt
```

### Run Verification Test Suite
```powershell
python scripts/verify_dashboard_data.py
```

### Launch Interactive Research Dashboard
```powershell
streamlit run dashboard/app.py
```
The application will launch on local port `8501` (e.g., `http://localhost:8501`).

---

## 8. Confirmation of Research Data Integrity

It is hereby confirmed that:
- No canonical dataset was modified or overwritten.
- No train/test splits were altered.
- No baseline or mitigated model definitions were changed.
- No SHAP calculations or explanation consistency formulas were altered.
- No authoritative result files in `results/` were modified or regenerated.
- All Phase 3, Phase 4, and Phase 5 frozen numerical values remain 100% intact.

---

## Final Status Determination

```text
================================================================================
PHASE 6 COMPLETE — FAIRCredit-AI DASHBOARD READY FOR REVIEW
================================================================================
```
