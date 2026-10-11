# Frontend Runtime Fix Report: Fairness & Research Routes

**Project:** FairCredit-AI: A Responsible AI Credit Model Auditing Framework  
**Repository Path:** `D:\sarthak\Subjects\ERA\FairCredit AI`  
**Date:** October 11, 2026  
**Status:** RESOLVED & VERIFIED  

---

## 1. Executive Summary

This report documents the diagnosis, root-cause analysis, code remediation, and full verification for two runtime crashes on the Next.js frontend:
1. **Bug 1 — `/fairness` (`r.dataset.toLowerCase()`):** `Cannot read properties of undefined (reading 'toLowerCase')` around line 83 in `frontend/src/app/fairness/page.tsx`.
2. **Bug 2 — `/research` (`row.model.replace(...)`):** `Cannot read properties of undefined (reading 'replace')` around line 296 in `frontend/src/app/research/page.tsx`.

Both issues were traced to a schema and data-source mismatch between the API endpoint `/api/research/matrix` and the frontend consumers. The endpoint previously loaded `fairness_performance_explanation_matrix.csv` (a 6-row qualitative markdown summary with capitalized headers `Dataset`, `Model` and qualitative description columns) instead of the comprehensive quantitative experimental results matrix.

The fix unified the data boundary in `backend/app/services/research_service.py` by joining `results/mitigation_master_results.csv` (18 empirical mitigation experiments) with `results/shap_summary.csv` (SHAP consistency metrics) and standardizing keys (`dataset`, `model`, `mitigation_method`, `baseline_acc`, etc.). Robust normalization and defensive formatters were added to both frontend pages and shared utility helpers.

---

## 2. Root Cause Analysis

### Bug 1: `/fairness`
- **Location:** `frontend/src/app/fairness/page.tsx:83`
- **Expression:** `r.dataset.toLowerCase() === selectedDataset.toLowerCase()`
- **Cause:**
  - `/api/research/matrix` was returning records parsed from `fairness_performance_explanation_matrix.csv`.
  - That CSV had keys `Dataset`, `Model`, `Fairness_Impact`, `Performance_Impact`, `Explanation_Impact` rather than `dataset`, `model`, `mitigation_method`.
  - In JavaScript, `r.dataset` evaluated to `undefined`, causing `.toLowerCase()` to throw a TypeError: `Cannot read properties of undefined (reading 'toLowerCase')`.

### Bug 2: `/research`
- **Location:** `frontend/src/app/research/page.tsx:296`
- **Expression:** `row.model.replace(/_/g, " ")`
- **Cause:**
  - For the same reason, `row.model` evaluated to `undefined` (because the CSV key was `Model`).
  - Attempting to call `.replace(...)` on `undefined` threw a TypeError: `Cannot read properties of undefined (reading 'replace')`.
  - Furthermore, all quantitative metric columns (`baseline_acc`, `mitigated_acc`, `baseline_auc`, `mitigated_auc`, `baseline_eod`, `mitigated_eod`, `shap_cosine_sim`, `shap_top5_overlap`) were missing from `fairness_performance_explanation_matrix.csv`, so the table could not display the authoritative empirical research results even if keys had been matched.

---

## 3. Remediation & Implementation Details

### A. Data-Boundary Fix: `backend/app/services/research_service.py`
1. Re-implemented `get_research_matrix()` to assemble the authoritative 18-row quantitative matrix from:
   - `results/mitigation_master_results.csv` (18 configurations across 2 datasets, 3 models, and 3 mitigation methods).
   - `results/shap_summary.csv` (SHAP explanation consistency metrics: Cosine Similarity, Top-5 Feature Overlap, Spearman Rank Correlation).
2. Standardized key normalization:
   - `dataset` (e.g. `taiwan`, `german`)
   - `model` (e.g. `logistic_regression`, `random_forest`, `xgboost`)
   - `mitigation_method` (e.g. `reweighing`, `exponentiated_gradient`, `threshold_optimizer`)
   - Numerical metrics: `baseline_acc`, `mitigated_acc`, `baseline_auc`, `mitigated_auc`, `baseline_eod`, `mitigated_eod`, `shap_cosine_sim`, `shap_top5_overlap`, `shap_spearman_rank`.
3. Created `get_qualitative_matrix()` to retain access to the qualitative synthesis matrix without corrupting the quantitative API.

### B. Route Addition: `backend/app/api/routes.py`
- Added `@router.get("/research/qualitative-matrix")` to cleanly serve the qualitative matrix to any components requesting it, maintaining clean separation of concerns.

### C. Shared Normalization Utilities: `frontend/src/lib/utils.ts`
- Added defensive helper functions:
  - `formatDatasetLabel(dataset)`: Safely formats dataset keys into human-readable strings (`Taiwan Default Credit`, `German Credit`), returning `"Unavailable"` if null/undefined.
  - `formatModelLabel(model)`: Safely formats model keys (`Logistic Regression`, `Random Forest`, `XGBoost`), returning `"Unavailable"` if null/undefined.
  - `formatMitigationLabel(mitigation)`: Safely formats mitigation technique keys (`Reweighing`, `Exponentiated Gradient`, `ThresholdOptimizer`), returning `"Unavailable"` if null/undefined.
- Enhanced `formatMetric`, `formatPercent`, and `formatDelta` to handle empty strings, `NaN`, `undefined`, and `null` gracefully.

### D. Frontend Pages: `/fairness` and `/research`
- **`frontend/src/app/fairness/page.tsx`**:
  - Implemented `normalizeMatrixRecord` and `normalizeShapSummaryRecord` to handle case variations (`Dataset` vs `dataset`) and normalize keys before use.
  - Added safe configuration matching: checks `r.dataset && r.model && r.mitigation_method` before calling `.toLowerCase()`.
  - Added graceful fallback empty states and informative missing-data notices rather than crashing the page.
- **`frontend/src/app/research/page.tsx`**:
  - Implemented `normalizeMatrixRecord` to normalize incoming data rows.
  - Replaced inline `.replace(...)` calls with `formatModelLabel(row.model)`, `formatDatasetLabel(row.dataset)`, and `formatMitigationLabel(row.mitigation_method)`.
  - Added empty/loading state handling with fallback messaging.

---

## 4. Research Integrity Protection

In accordance with strict research integrity constraints:
- **No model retraining or weight modification** was performed.
- **No frozen CSVs or metrics** in `results/` or `data/` were modified or recomputed.
- **No synthetic data or fabricated records** were added.
- The 18 quantitative rows displayed are direct, 1-to-1 reflections of `results/mitigation_master_results.csv` and `results/shap_summary.csv`.

---

## 5. Verification Results

### A. Next.js Production Build
```bash
npm run build (in frontend/)
# Compiled in 963ms
# Route (app)                              Size     First Load JS
#  /                                    184 B           101 kB
#  /_not-found                           993 B           102 kB
#  /assess                              201 kB           302 kB
#  /fairness                            200 kB           301 kB
#  /research                            200 kB           301 kB
# Result: SUCCESS (0 errors, 0 warnings)
```

### B. Backend API Test Suite
```bash
python -m pytest backend/tests/test_api.py -v
# 8 passed in 1.48s
# Result: 100% PASS
```

### C. Live Route Probing (`scripts/test_rendered_pages.py`)
All four application routes were tested against the live server:
- `http://localhost:3000/`  HTTP 200, Content Verified (FairCredit-AI, Default Probability)
- `http://localhost:3000/assess`  HTTP 200, Content Verified (Credit-Risk Assessment)
- `http://localhost:3000/fairness`  HTTP 200, Content Verified (Mitigation Audit & Fair Credit Scoring)
- `http://localhost:3000/research`  HTTP 200, Content Verified (Empirical Research Matrix)
- Verified all 18 mitigation experiments render in `/research`.
- Verified selector configurations work across Taiwan & German datasets, 3 models, and all mitigation methods.

### D. Authoritative Research Verification Scripts
Executed from repository root using `.venv`:
1. `python scripts/verify_baseline_sanity.py`  **PASS** (Taiwan & German ROC-AUC sanity verified)
2. `python scripts/verify_mitigation_results.py`  **PASS** (All 18 mitigation rows, metric bounds, and EOD improvements verified)
3. `python scripts/verify_dashboard_data.py`  **PASS** (Master results, subgroup metrics, and SHAP summary schema verified)
4. `python scripts/verify_shap_consistency.py`  **PASS** (SHAP explanation consistency, cosine similarity, rank correlations verified)

---

## 6. Files Changed

| File Path | Description of Change |
|---|---|
| `backend/app/services/research_service.py` | Assembles quantitative matrix from master CSVs and normalizes field keys. Added `get_qualitative_matrix`. |
| `backend/app/api/routes.py` | Added `/research/qualitative-matrix` route. |
| `frontend/src/lib/utils.ts` | Added `formatDatasetLabel`, `formatModelLabel`, `formatMitigationLabel`, and defensive NaN handling. |
| `frontend/src/app/fairness/page.tsx` | Added record normalization, safe selector matching, and graceful empty states. |
| `frontend/src/app/research/page.tsx` | Added record normalization, safe label formatters, and table empty states. |
| `scripts/test_rendered_pages.py` | Automated verification script testing all 4 Next.js routes and backend APIs. |
| `reports/FRONTEND_RUNTIME_FIX_REPORT.md` | Comprehensive documentation of root cause, fixes, and verification. |
