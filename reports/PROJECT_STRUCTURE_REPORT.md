# Project Structure & Architecture Report: FairCredit-AI

**Project:** FairCredit-AI: A Responsible AI Credit Model Auditing Framework  
**Repository Path:** `D:\sarthak\Subjects\ERA\FairCredit AI`  
**Date:** October 11, 2026  
**Status:** REORGANIZED, VERIFIED & PRODUCTION READY  

---

## 1. Structural Overview & Separation of Concerns

To prepare FairCredit-AI for enterprise research presentation and full-stack maintainability, the project directory was organized according to clean architectural boundaries:

```text
FairCredit AI/
├── frontend/                     # Next.js 16 Web Application (Presentation Layer)
│   ├── src/
│   │   ├── app/                  # App Router pages (/ , /assess, /fairness, /research)
│   │   ├── components/           # Reusable UI components (Navbar, Footer, MetricCard, ShapBarChart)
│   │   ├── lib/                  # API client, utility functions, formatting
│   │   └── types/                # TypeScript shared types
│   ├── package.json              # Isolated frontend npm manifest
│   └── tsconfig.json
├── backend/                      # FastAPI Python Services (Inference & Data Layer)
│   ├── app/
│   │   ├── api/routes.py         # REST endpoints (/health, /predict, /explain, /research/*)
│   │   ├── schemas/              # Pydantic v2 validation models
│   │   ├── services/             # Model loading, live inference, SHAP, and research loaders
│   │   └── main.py               # Application entrypoint & CORS middleware
│   └── tests/test_api.py         # Automated API tests
├── models/                       # Validated Model Artifacts
│   ├── models_manifest.json      # Dataset schemas, ranges, demo presets
│   └── *.joblib                  # Serialized pipelines (Baseline, Reweighed, Backgrounds)
├── src/                          # Core Research ML Pipeline (Intact for scripts)
├── scripts/                      # Reproducible verification & artifact generation scripts
├── data/                         # Canonical raw and processed datasets
├── results/                      # Immutable frozen research outputs (CSVs, JSONs, figures)
├── dashboard/                    # Retained Streamlit prototype (legacy fallback)
├── reports/                      # Full auditing and migration reports
├── requirements.txt              # Python dependency manifest
├── .gitignore                    # Clean ignores (excludes .venv, node_modules, .next, caches)
└── README.md                     # Comprehensive project documentation
```

---

## 2. Key Architectural Decisions & Rationales

1. **Strict Separation of Frontend and Backend:**
   - Frontend dependencies live exclusively in `frontend/package.json`.
   - Backend and ML dependencies live in `requirements.txt`.
   - No conflicting manifests or combined mono-dependency manifests.

2. **Preservation of Root `src/`:**
   - To guarantee that existing verification suites (`verify_baseline_sanity.py`, `verify_mitigation_results.py`, `verify_shap_consistency.py`, `verify_dashboard_data.py`) continue to execute with 100% zero path breakage, `src/` was preserved at the root.

3. **Inference Pipeline & Serialization:**
   - The interactive credit-risk assessment performs genuine live ML inference via FastAPI.
   - `scripts/export_inference_artifacts.py` deterministically trains and serializes Baseline and Reweighed models with `test_size=0.20, random_state=42`.
   - All 6 configurations reproduce the exact four-decimal test set ROC-AUC and Accuracy recorded in authoritative research files.

4. **Live Local SHAP Explainability:**
   - Instead of mocked attributions, the FastAPI service computes genuine SHAP feature values for newly scored applicants using `LinearExplainer` (for Logistic Regression) and `TreeExplainer` (for Random Forest and XGBoost), with XGBoost 3.x base_score UBJSON compatibility patch.

5. **Legacy Streamlit Dashboard Retained:**
   - `dashboard/` was kept intact as a secondary legacy verification interface without colliding with Next.js.

---

## 3. Verification Summary

- **Frontend Production Build:** `npm run build` completed with 0 errors (100% static routes generated).
- **Backend Unit Tests:** `pytest backend/tests/test_api.py` passed (8 of 8 tests).
- **Core Research Scripts:** All 4 verification suites passed 100%.
- **Live HTTP Probes:** All 8 frontend and backend endpoints returned HTTP 200 OK.
