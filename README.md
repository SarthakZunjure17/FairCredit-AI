# FairCredit-AI: A Responsible AI Credit Model Auditing Framework

**Tagline:** Evaluating Fairness, Performance, and Explanation Consistency in Credit Scoring Models  
**Institution:** Department of Computer Science and AI, Vishwakarma Institute of Technology, Pune  
**Academic Year:** 2026–2027  
**Research Status:** `LOCKED & FROZEN` (Validated, Remediated, Independently Audited)  

---

> ### Regulatory & Auditing Notice
> **Research Prototype — Model Auditing & Responsible AI Decision Support**  
> **Not for Automated Credit Decisions.** This software is an empirical auditing framework and decision-support prototype designed for model-risk analysts and researchers. It does **not** determine loan eligibility, evaluate real credit applicants, or operate as an automated underwriting or compliance engine.

---

## 1. Quick Start: Next.js + FastAPI Platform

FairCredit-AI features an integrated full-stack architecture combining interactive credit-risk assessment with frozen empirical research auditing:

### Terminal 1: Launch FastAPI Backend (Port 8000)
```powershell
cd "D:\sarthak\Subjects\ERA\FairCredit AI"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Endpoint: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### Terminal 2: Launch Next.js Frontend (Port 3000)
```powershell
cd "D:\sarthak\Subjects\ERA\FairCredit AI\frontend"
npm run dev -- -p 3000
```
- Web Application: `http://localhost:3000`

---

## 2. Research Question & Objective

In algorithmic credit underwriting, financial institutions are increasingly required to ensure algorithmic fairness while providing individualized explanations (e.g., adverse action notices under ECOA / FCRA). While fairness mitigations alter model parameters or decision boundaries, their downstream impact on feature attributions has historically remained under-explored.

**Core Research Question:**  
> *"When fairness mitigation is applied to machine-learning credit-scoring models, how does it affect predictive performance, group fairness, and explanation consistency?"*

FairCredit-AI systematically audits credit-scoring pipelines across four core pillars:
1. **Predictive Performance:** Accuracy, Balanced Accuracy, Precision, Recall, F1, and continuous ROC-AUC.
2. **Group Fairness:** Equalized Odds Difference (EOD), Equal Opportunity Difference (EOpp), Demographic Parity Difference (DPD), and Approval Disparate Impact (DI).
3. **Local Explainability:** Feature-level SHAP attributions toward default risk ($Y=1$).
4. **Explanation Consistency:** Multi-metric stability of feature attributions across identical borrower cohorts post-mitigation.

---

## 3. Benchmark Datasets & Preprocessing

The framework audits two canonical credit benchmarks under strict data leakage controls:

| Dataset | Sample Size ($N$) | Predictive Features ($P$) | Default Prevalence ($Y=1$) | Primary Protected Attribute | Evaluation Split |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Taiwan Credit Card Default** | 30,000 | 23 financial variables | 22.12% | `sex` (Female: 60.4%, Male: 39.6%) | Stratified 80/20 ($N_{\text{test}}=6,000$) |
| **Statlog German Credit** | 1,000 | 20 credit attributes | 30.00% | `derived_sex` (Female: 31.0%, Male: 69.0%) | Stratified 80/20 ($N_{\text{test}}=200$) |

### Leakage and Governance Controls
- **Protected Attribute Isolation:** Protected demographic attributes (`sex`, `derived_sex`) are **strictly excluded** from the model feature matrix $X$. Models learn exclusively from predictive financial and credit history features. Demographic labels are supplied separately to Fairlearn evaluators.
- **Preprocessing Isolation:** All imputers, encoders, and scalers are fitted strictly on the training partitions (`random_state=42`). Test partitions are evaluated strictly out-of-sample.

---

## 4. Modeling & Fairness Mitigation Methodology

### Model Architectures Under Study
1. **Logistic Regression (`L2` penalty):** Linear baseline with convex optimization.
2. **Random Forest (`n_estimators=100`, unconstrained depth):** Bagged non-linear tree ensemble.
3. **XGBoost (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`):** Regularized gradient boosted decision trees.

### Three Levels of Fairness Mitigation
1. **Pre-processing (Reweighing):**
   - Modifies training sample weights according to the joint distribution of protected status and default outcome: $W(S, Y) = \frac{P(S)P(Y)}{P(S, Y)}$.
   - Yields a genuinely refitted continuous scoring model.
   - **Primary focus of SHAP explanation-consistency analysis.**
2. **In-processing (Exponentiated Gradient):**
   - Solves a min-max game yielding a randomized ensemble of predictors subject to Equalized Odds constraints.
   - Outputs binary/step decisions; continuous ranking scores are reported strictly as **N/A**.
3. **Post-processing (ThresholdOptimizer):**
   - Derives group-specific decision thresholds on baseline continuous scores to equalize TPR and FPR.
   - Leaves the underlying baseline continuous scoring function unchanged.

---

## 5. SHAP Explanation Consistency Protocol

Local explanations are computed using exact SHAP explainers aligned to model families:
- **Logistic Regression:** `LinearExplainer` (margin / log-odds space) using training-derived background data ($N_{\text{bg}}=100$ for German, $N_{\text{bg}}=200$ for Taiwan).
- **Random Forest:** `TreeExplainer` explaining class-1 positive default probability $P(Y=1|X)$.
- **XGBoost:** `TreeExplainer` explaining log-odds margin space.

### Explanation Consistency Metrics
1. **Cosine Similarity:** Measures directional attribution alignment in multidimensional space.
2. **Top-5 Feature Overlap:** Jaccard-style overlap of the 5 highest-impact decision drivers.
3. **Spearman Rank Correlation:** Monotonic rank preservation across all features.
4. **95% Bootstrap Confidence Intervals:** Computed over 1,000 bootstrap iterations (`random_state=42`).

---

## 6. Frozen Multi-Objective Research Matrix

Empirical synthesis of the 6 primary Baseline vs Reweighing configurations:

| Dataset | Model | ΔAUC | ΔEOD | ΔDPD | Mean SHAP Cosine [95% CI] | Mean Top-5 Overlap | Mean Spearman ρ | Observed Profile |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Taiwan** | Logistic Regression | +0.0008 | +0.0035 | 0.0000 | 0.9996 [0.9995, 0.9996] | 0.9800 | 0.9832 | High Consistency, Minimal Fairness Shift |
| **Taiwan** | Random Forest | +0.0020 | -0.0079 | -0.0018 | 0.9380 [0.9341, 0.9419] | 0.7924 | 0.7768 | Fairness Improved, Moderate Consistency |
| **Taiwan** | XGBoost | -0.0011 | +0.0130 | -0.0024 | 0.9787 [0.9772, 0.9803] | 0.8864 | 0.8871 | High Consistency, Slight EOD Worsened |
| **German** | Logistic Regression | -0.0006 | +0.0500 | -0.0446 | 0.9983 [0.9982, 0.9984] | 0.9540 | 0.9908 | DPD Improved, EOD Worsened |
| **German** | Random Forest | +0.0137 | -0.0750 | -0.0615 | 0.9652 [0.9624, 0.9677] | 0.8370 | 0.8891 | **Multi-Objective Win (Fairness & AUC Improved)** |
| **German** | XGBoost | -0.0101 | 0.0000 | +0.0052 | 0.9757 [0.9735, 0.9779] | 0.8380 | 0.9173 | Invariant Fairness, High Consistency |

---

## 7. Repository Structure

```text
FairCredit AI/
├── frontend/                     # Next.js 16 Web Application (App Router, TS, Tailwind, Recharts)
├── backend/                      # FastAPI Python Services (REST Endpoints, Inference, SHAP)
├── models/                       # Serialized trained model pipelines & manifest
├── src/                          # Research ML source (modeling, explainability, consistency)
├── scripts/                      # Reproducible verification & artifact generation scripts
├── data/                         # Canonical raw and processed datasets
├── results/                      # Frozen experiment result tables & figures
├── dashboard/                    # Retained Streamlit prototype (legacy fallback)
├── reports/                      # Auditing and migration reports
├── requirements.txt              # Python dependency manifest
├── .gitignore                    # Clean Git ignores
└── README.md                     # Master documentation
```

---

## 8. Verification & Test Commands

Run the full battery of tests to confirm complete project reproducibility:

```powershell
# 1. Execute FastAPI backend unit tests (8 tests)
.\.venv\Scripts\python.exe -m pytest backend/tests/test_api.py -v

# 2. Execute Next.js production build check
cd frontend
npm run build
cd ..

# 3. Execute baseline reconstruction sanity suite
.\.venv\Scripts\python.exe scripts\verify_baseline_sanity.py

# 4. Execute mitigation results verification (all 18 experiments)
.\.venv\Scripts\python.exe scripts\verify_mitigation_results.py

# 5. Execute dashboard data verification suite (10 checks)
.\.venv\Scripts\python.exe scripts\verify_dashboard_data.py

# 6. Execute SHAP explanation consistency audit (12 checks)
.\.venv\Scripts\python.exe scripts\verify_shap_consistency.py
```
