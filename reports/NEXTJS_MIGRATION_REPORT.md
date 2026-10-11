# Next.js Full-Stack Migration Report: FairCredit-AI

**Project:** FairCredit-AI: A Responsible AI Credit Model Auditing Framework  
**Repository Path:** `D:\sarthak\Subjects\ERA\FairCredit AI`  
**Date:** October 11, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

The FairCredit-AI research prototype has been migrated to an integrated, modern full-stack web application combining:
- **Experience A (Credit Risk Assessment):** Interactive applicant scoring on verified trained machine-learning models, comparing Baseline vs. Reweighed default probabilities, and computing live local SHAP feature attributions.
- **Experience B (Fairness Audit & Research Workspace):** Auditing empirical group fairness metrics (Equalized Odds, Demographic Parity) across pre-, in-, and post-processing mitigations, and evaluating explanation stability (Cosine Similarity, Top-5 Overlap, Spearman Rank Correlation).

All authoritative research findings, frozen results (`results/`), and original validation suites remain 100% intact and verified.

---

## 2. Technology Stack & Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Next.js 16 (App Router)                     │
│  - React 19 + TypeScript (Strict Type Checking)                 │
│  - Tailwind CSS v4 + Recharts Visualization                     │
│  - Lucide Icons + Minimalist Fintech Design System              │
│  - 4 Destinations: Overview, Assess, Fairness, Research         │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP JSON API (Port 8000)
┌────────────────────────────────▼────────────────────────────────┐
│                         FastAPI Backend                         │
│  - REST Endpoints (/health, /configurations, /predict, /explain)│
│  - Pydantic v2 Request/Response Validation                      │
│  - Real Preprocessing & Live Model Inference (scikit-learn, XGB)│
│  - Live SHAP Attribution (LinearExplainer, TreeExplainer)       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│            Serialized Artifacts & Research Source of Truth       │
│  - models/ (12 verified Baseline & Reweighed pipelines)         │
│  - results/ (Authoritative frozen CSVs, JSONs, matrices)         │
│  - dashboard/data/ (2,100 applicant held-out SHAP Parquet)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```text
FairCredit AI/
├── frontend/                     # Next.js 16 Web Application
│   ├── src/
│   │   ├── app/                  # App Router pages (/ , /assess, /fairness, /research)
│   │   │   ├── layout.tsx        # Global Layout with Navbar & Footer
│   │   │   ├── page.tsx          # Page 1: Overview
│   │   │   ├── assess/page.tsx   # Page 2: Assess Risk
│   │   │   ├── fairness/page.tsx # Page 3: Fairness & Explanations
│   │   │   └── research/page.tsx # Page 4: Research Synthesis
│   │   ├── components/           # Navbar, Footer, MetricCard, ShapBarChart
│   │   ├── lib/                  # api client, utils, formatters
│   │   └── types/                # TypeScript interface definitions
│   ├── package.json
│   └── tsconfig.json
├── backend/                      # FastAPI Python Backend
│   ├── app/
│   │   ├── api/routes.py         # Endpoints (/health, /predict, /explain, /research/*)
│   │   ├── schemas/              # Pydantic models (assessment.py, research.py)
│   │   ├── services/             # model_service.py (inference & SHAP), research_service.py
│   │   └── main.py               # FastAPI entrypoint with CORS
│   └── tests/test_api.py         # Automated API & Inference test battery
├── models/                       # Serialized trained model artifacts & manifest
│   ├── models_manifest.json      # Dataset schemas, ranges, sample profiles
│   └── *.joblib                  # Serialized pipelines (baseline, reweighed, backgrounds)
├── src/                          # Original research ML source (modeling, explainability)
├── scripts/                      # Reproducible verification & artifact export scripts
├── data/                         # Raw & processed canonical datasets
├── results/                      # Frozen experiment result tables & figures
├── dashboard/                    # Legacy Streamlit dashboard & Parquet data
└── reports/                      # Project audit & migration documentation
```

---

## 4. Model Configurations & Inference Verification

All 6 configurations were serialized via `scripts/export_inference_artifacts.py` using fixed `random_state=42` and identical `test_size=0.20` stratified splits. Test set metrics match authoritative frozen results to four decimal places:

| Dataset | Model Architecture | Baseline ROC-AUC | Reweighed ROC-AUC | Baseline Acc | Reweighed Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Taiwan** | Logistic Regression | 0.7574 | 0.7582 | 0.8173 | 0.8175 |
| **Taiwan** | Random Forest | 0.7528 | 0.7548 | 0.8118 | 0.8123 |
| **Taiwan** | XGBoost | 0.7785 | 0.7774 | 0.8193 | 0.8190 |
| **German** | Logistic Regression | 0.8061 | 0.8055 | 0.7850 | 0.7850 |
| **German** | Random Forest | 0.7798 | 0.7935 | 0.7800 | 0.7800 |
| **German** | XGBoost | 0.8019 | 0.7918 | 0.7750 | 0.7750 |

---

## 5. Verification Suites Executed

1. **Next.js Production Build (`npm run build`):**
   - Result: **PASS** (Compiled in 7.4s, 0 TypeScript errors, 100% static routes generated).
2. **FastAPI Backend Tests (`pytest backend/tests/test_api.py`):**
   - Result: **8 passed in 5.36s** (Health, configurations, prediction, live SHAP explanation, research data).
3. **Phase 1 Baseline Sanity Suite (`scripts/verify_baseline_sanity.py`):**
   - Result: **PASS** (Exit code 0).
4. **Phase 3 Mitigation Results Suite (`scripts/verify_mitigation_results.py`):**
   - Result: **PASS** (18 experiments verified, 0 data leakage, exit code 0).
5. **Phase 6 Dashboard Data Suite (`scripts/verify_dashboard_data.py`):**
   - Result: **PASS** (All 10 checks verified, exit code 0).
6. **Phase 4 SHAP Consistency Suite (`scripts/verify_shap_consistency.py`):**
   - Result: **PASS** (All 12 checks verified, exit code 0).

---

## 6. How to Run Locally

### Terminal 1: Start FastAPI Backend
```powershell
cd "D:\sarthak\Subjects\ERA\FairCredit AI"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
URL: `http://127.0.0.1:8000` (Docs: `http://127.0.0.1:8000/docs`)

### Terminal 2: Start Next.js Frontend
```powershell
cd "D:\sarthak\Subjects\ERA\FairCredit AI\frontend"
npm run dev -- -p 3000
```
URL: `http://localhost:3000`
