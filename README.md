# FairCredit-AI: A Responsible AI Credit Model Auditing Framework

**Tagline:** Evaluating Fairness, Performance, and Explanation Consistency in Credit Scoring Models  
**Institution:** Department of Computer Science and AI, Vishwakarma Institute of Technology, Pune  
**Academic Year:** 2026–2027  
**Research Status:** `LOCKED & FROZEN` (Validated, Remediated, Independently Audited)  

---

> ### ⚖️ Regulatory & Auditing Notice
> **Research Prototype — Model Auditing & Responsible AI Decision Support**  
> **Not for Automated Credit Decisions.** This software is an empirical auditing framework and decision-support prototype designed for model-risk analysts and researchers. It does **not** determine loan eligibility, evaluate real credit applicants, or operate as an automated underwriting or compliance engine. No automated loan decisioning buttons are provided.

---

## 1. Research Question & Objective

In algorithmic credit underwriting, financial institutions are increasingly required to ensure algorithmic fairness while providing individualized explanations (e.g., adverse action notices under ECOA / FCRA). While fairness mitigations alter model parameters or decision boundaries, their downstream impact on feature attributions has historically remained under-explored.

**Core Research Question:**  
> *"When fairness mitigation is applied to machine-learning credit-scoring models, how does it affect predictive performance, group fairness, and explanation consistency?"*

FairCredit-AI systematically audits credit-scoring pipelines across four core pillars:
1. **Predictive Performance:** Accuracy, Balanced Accuracy, Precision, Recall, F1, and continuous ROC-AUC.
2. **Group Fairness:** Equalized Odds Difference (EOD), Equal Opportunity Difference (EOpp), Demographic Parity Difference (DPD), and Approval Disparate Impact (DI).
3. **Local Explainability:** Feature-level SHAP attributions toward default risk ($Y=1$).
4. **Explanation Consistency:** Multi-metric stability of feature attributions across identical borrower cohorts post-mitigation.

---

## 2. Benchmark Datasets & Preprocessing

The framework audits two canonical credit benchmarks under strict data leakage controls:

| Dataset | Sample Size ($N$) | Predictive Features ($P$) | Default Prevalence ($Y=1$) | Primary Protected Attribute | Evaluation Split |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Taiwan Credit Card Default** | 30,000 | 23 financial variables | 22.12% | `sex` (Female: 60.4%, Male: 39.6%) | Stratified 80/20 ($N_{\text{test}}=6,000$) |
| **Statlog German Credit** | 1,000 | 20 credit attributes | 30.00% | `derived_sex` (Female: 31.0%, Male: 69.0%) | Stratified 80/20 ($N_{\text{test}}=200$) |

### Leakage and Governance Controls
- **Protected Attribute Isolation:** Protected demographic attributes (`sex`, `derived_sex`) are **strictly excluded** from the model feature matrix $X$. Models learn exclusively from predictive financial and credit history features. Demographic labels are supplied separately to Fairlearn evaluators.
- **Preprocessing Isolation:** All imputers, encoders, and scalers are fitted strictly on the training partitions (`random_state=42`). Test partitions are evaluated strictly out-of-sample.

---

## 3. Modeling & Fairness Mitigation Methodology

### Model Architectures Under Study
1. **Logistic Regression (`L2` penalty):** Linear baseline with convex optimization.
2. **Random Forest (`n_estimators=100`, `max_depth=10`):** Bagged non-linear tree ensemble.
3. **XGBoost (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`):** Regularized gradient boosted decision trees.

### Three Levels of Fairness Mitigation
1. **Pre-processing (Reweighing):**
   - Modifies training sample weights according to the joint distribution of protected status and default outcome: $W(S, Y) = \frac{P(S)P(Y)}{P(S, Y)}$.
   - Yields a genuinely refitted continuous scoring model.
   - **Primary focus of SHAP explanation-consistency analysis.**
2. **In-processing (Exponentiated Gradient):**
   - Solves a min-max game yielding a randomized ensemble of predictors subject to Equalized Odds constraints.
   - Outputs binary/step decisions; unthresholded continuous ranking scores are reported strictly as **N/A**.
   - Excluded from primary SHAP scoring consistency due to lack of a continuous scoring function.
3. **Post-processing (ThresholdOptimizer):**
   - Derives group-specific decision thresholds on baseline continuous scores to equalize TPR and FPR.
   - Leaves the underlying baseline continuous scoring function unchanged.
   - Excluded from primary SHAP scoring consistency because the underlying scoring model is not modified.

---

## 4. SHAP Explanation Consistency Protocol

Local explanations are computed using exact SHAP explainers aligned to model families:
- **Logistic Regression:** `LinearExplainer` (margin / log-odds space) using training-derived background data ($N_{\text{bg}}=100$ for German, $N_{\text{bg}}=200$ for Taiwan).
- **Random Forest:** `TreeExplainer` explaining class-1 positive default probability $P(Y=1|X)$.
- **XGBoost:** `TreeExplainer` explaining log-odds margin space.

### Evaluated Cohorts
- **German Credit:** All $N=200$ test applicants ($100\%$ evaluation).
- **Taiwan Credit:** Deterministic first $N=500$ test applicants.
- **Total Individual Explanations:** 2,100 applicant explanations across the 6 primary pairs.

### Explanation Consistency Metrics
1. **Cosine Similarity:** Measures directional attribution alignment in multidimensional space.
2. **Top-5 Feature Overlap:** Jaccard-style overlap of the 5 highest-impact decision drivers.
3. **Spearman Rank Correlation:** Monotonic rank preservation across all features.
4. **95% Bootstrap Confidence Intervals:** Computed over 1,000 bootstrap iterations (`random_state=42`).

---

## 5. Frozen Multi-Objective Research Matrix

Empirical synthesis of the 6 primary Baseline vs Reweighing configurations:

| Dataset | Model | $\Delta$AUC | $\Delta$EOD | $\Delta$DPD | Mean SHAP Cosine [95% CI] | Mean Top-5 Overlap | Mean Spearman $\rho$ | Observed Profile |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Taiwan** | Logistic Regression | +0.0008 | +0.0035 | 0.0000 | 0.9996 [0.9995, 0.9996] | 0.9800 | 0.9832 | Fairness Divergence (EOD Worsened, High Consistency) |
| **Taiwan** | Random Forest | +0.0020 | -0.0079 | -0.0018 | 0.9380 [0.9341, 0.9419] | 0.7924 | 0.7768 | Fairness Improved, Stable AUC, Moderate Consistency |
| **Taiwan** | XGBoost | -0.0011 | +0.0130 | -0.0024 | 0.9787 [0.9772, 0.9803] | 0.8864 | 0.8871 | Fairness Worsened, Stable AUC, High Consistency |
| **German** | Logistic Regression | -0.0006 | +0.0500 | -0.0446 | 0.9983 [0.9982, 0.9984] | 0.9540 | 0.9908 | Fairness Divergence (EOD Worsened, DPD Improved) |
| **German** | Random Forest | +0.0137 | -0.0750 | -0.0615 | 0.9652 [0.9624, 0.9677] | 0.8370 | 0.8891 | **Multi-Objective Win (Fairness & AUC Improved)** |
| **German** | XGBoost | -0.0101 | 0.0000 | +0.0052 | 0.9757 [0.9735, 0.9779] | 0.8380 | 0.9173 | Invariant Fairness, Slight AUC Drop, High Consistency |

### Core Empirical Findings
- **Central Synthesis Finding:** Fairness mitigation through Reweighing exhibited heterogeneous effects on explanation consistency across dataset-model configurations rather than a universal fairness–explanation trade-off.
- **Architectural Hierarchy (Observed):**
  - *Logistic Regression:* Near-perfect consistency (Cosine $> 0.998$) due to convex parameter stability.
  - *XGBoost:* High consistency (Cosine $0.975 - 0.979$) due to regularized additive boosting.
  - *Random Forest:* High directional consistency (Cosine $0.938 - 0.965$), but with greater feature-ranking variability (Top-5 overlap drops to $0.792 - 0.837$).
- **Multi-Dimensional Fairness Divergence:** Improvement in demographic-parity-related metrics does not guarantee improvement in Equalized Odds (as observed in Taiwan XGBoost and German Logistic Regression).

---

## 6. Repository Architecture

```text
project_root/
├── dashboard/                               # Streamlit Model-Risk Dashboard
│   ├── app.py                              # Master interactive audit interface
│   ├── data_loader.py                      # Programmatic frozen CSV/Parquet loader
│   ├── components.py                       # Metric cards, disclaimers, charts
│   └── data/
│       └── individual_feature_shap.parquet # Frozen individual SHAP attributions (167,700 rows)
│
├── results/                                 # Authoritative Frozen Research Artifacts
│   ├── mitigation_master_results.csv       # Complete 18-experiment mitigation audit
│   ├── final_research_results.csv          # 6 primary baseline vs reweighing rows
│   ├── shap_summary.csv                    # Aggregate explanation consistency metrics & CIs
│   ├── shap_group_metrics.csv              # Subgroup-stratified explanation metrics
│   ├── shap_individual_metrics.csv         # 2,100 applicant consistency records
│   ├── fairness_performance_explanation_matrix.csv # Cross-objective synthesis
│   ├── FINAL_RESEARCH_FREEZE.md            # Official research results lock document
│   └── figures/                            # Publication-grade figures (Figures 1-5)
│
├── scripts/                                 # Executable Research & Verification Scripts
│   ├── verify_dashboard_data.py            # Phase 6 data integrity test suite (10 checks)
│   ├── run_shap_analysis.py                # Phase 4 SHAP consistency pipeline
│   ├── compute_synthesis.py                # Phase 5 cross-objective synthesis
│   ├── verify_shap_consistency.py          # Phase 4.1 audit test suite (12 checks)
│   └── verify_mitigation_results.py        # Phase 3.1 independent reconstruction suite
│
├── src/                                     # Core Reusable Library
│   ├── explainability.py                   # SHAP explainer factory & background sampler
│   ├── consistency.py                      # Cosine, Top-K, Spearman, Bootstrap CIs
│   └── modeling/
│       ├── preprocessing.py                # Data loading & leakage-free pipelines
│       ├── models.py                       # Model factory
│       ├── mitigation.py                   # Reweighing, ExponentiatedGradient, ThresholdOptimizer
│       └── evaluation.py                   # Fairness & performance evaluators
│
├── requirements.txt                         # Python dependencies
└── README.md                                # Master documentation
```

---

## 7. How to Run the Dashboard

### Prerequisites
Install dependencies into your Python 3.10+ virtual environment:
```powershell
pip install -r requirements.txt
```

### Launch Interactive Research Dashboard
Start the Streamlit research prototype:
```powershell
streamlit run dashboard/app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### Run Automated Data & Dashboard Verification Suite
Before deployment or review, verify the 10 data integrity and schema checks:
```powershell
python scripts/verify_dashboard_data.py
```
This suite verifies:
1. All required authoritative source files exist.
2. The 6 primary Reweighing research rows are present.
3. Phase 3 metrics match authoritative master files.
4. Phase 4 consistency metrics match authoritative summaries.
5. Continuous ROC-AUC is never converted to zero for non-continuous mitigations (`N/A` strictly preserved).
6. Datasets are correctly identified across all tables.
7. All 3 model architectures are present.
8. Individual SHAP records have valid applicant IDs ($N=200$ German, $N=500$ Taiwan).
9. Zero leakage of protected demographic attributes into model feature matrices.
10. Dashboard data loader runs cleanly without schema or parsing errors.

---

## 8. Stated Research Limitations

1. **Benchmark Datasets:** Evaluated on two canonical public credit datasets (Taiwan Credit Card Default and Statlog German Credit).
2. **Fixed Split Policy:** Results reflect deterministic 80/20 stratified splits (`random_state=42`).
3. **Evaluation Cohorts:** Local explanation consistency is evaluated on $N=500$ applicants for Taiwan and $N=200$ applicants for German.
4. **German Sample Size:** German dataset is relatively small ($N=1,000$ total), creating wider confidence intervals.
5. **Mitigation Scope for SHAP:** Primary explanation-consistency experiment focuses on Reweighing.
6. **Post-Processing Exclusion:** ThresholdOptimizer operates purely as decision-level thresholding and does not modify the underlying continuous scoring function.
7. **In-Processing Exclusion:** ExponentiatedGradient outputs randomized predictor ensembles without a straightforward continuous scoring function under the locked evaluation protocol.
8. **Exploratory Associations:** Cross-objective associations cannot establish causal mechanisms.
9. **Generalizability:** Empirical findings reflect the audited datasets and model hyperparameters; they may not generalize to alternative lending environments or production banking portfolios.

---

## 9. Citation & Contact

If using FairCredit-AI in academic or regulatory research:
```bibtex
@article{faircredit_ai_2027,
  title={Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring},
  author={FairCredit-AI Research Team},
  institution={Department of Computer Science and AI, Vishwakarma Institute of Technology, Pune},
  year={2027}
}
```
