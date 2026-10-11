# Comprehensive Fairness & Performance Metric Audit

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Audit Component:** Section 6, 7, 8, 9, 10, 11 — Fairness Metrics, Data Leakage, Reproducibility & Independent Recomputation  
**Auditor:** Lead ML Data Engineer & Research Methodology Auditor  
**Date:** October 2026  
**Status:** **AUDIT PASSED FOR DISCRETE METRICS; CALCULATION MECHANICS 100% REPRODUCIBLE**

---

## 1. Audit Scope & Objectives

This audit provides a strict, independent verification of all mathematical definitions, code implementations, data flows, and empirical values across the Phase 3 fairness mitigation benchmark.

We verified:
1. Exact mathematical definitions of performance and fairness metrics.
2. Target convention alignment and directionality.
3. Subgroup confusion-matrix completeness and consistency.
4. Independent reconstruction of all 18 experiments (tolerance: $10^{-4}$).
5. Data leakage controls, feature isolation, and split reproducibility.

---

## 2. Metric Definitions & Code Implementations

### 2.1 Target Convention
Across all datasets and models, the target variable is strictly standardized as:
- **$Y = 1$:** Credit Default / Bad Credit Risk Event (Unfavorable ground-truth event).
- **$Y = 0$:** Non-Default / Good Credit (Favorable ground-truth event).

The model predictions follow:
- **$\hat{Y} = 1$:** Positive class prediction $\to$ Predicted Default / Rejection.
- **$\hat{Y} = 0$:** Negative class prediction $\to$ Predicted Non-Default / Favorable Credit Approval.

### 2.2 Confusion Matrix Quantities
For each sensitive demographic subgroup $g$:
- **$\text{TP}_g$:** Number of true defaulters correctly flagged as default ($Y=1, \hat{Y}=1$).
- **$\text{FP}_g$:** Number of non-defaulters incorrectly flagged as default ($Y=0, \hat{Y}=1$).
- **$\text{TN}_g$:** Number of non-defaulters correctly approved ($Y=0, \hat{Y}=0$).
- **$\text{FN}_g$:** Number of true defaulters erroneously approved ($Y=1, \hat{Y}=0$).
- **$N_g = \text{TP}_g + \text{TN}_g + \text{FP}_g + \text{FN}_g$:** Subgroup sample size.

Subgroup operational rates:
- **True Positive Rate (Recall on Defaulters):**
  $$\text{TPR}_g = \frac{\text{TP}_g}{\text{TP}_g + \text{FN}_g} = P(\hat{Y}=1 \mid Y=1, A=g)$$
- **False Positive Rate:**
  $$\text{FPR}_g = \frac{\text{FP}_g}{\text{FP}_g + \text{TN}_g} = P(\hat{Y}=1 \mid Y=0, A=g)$$
- **Default Prediction Rate (Selection Rate):**
  $$\text{SR}_g = \frac{\text{TP}_g + \text{FP}_g}{N_g} = P(\hat{Y}=1 \mid A=g)$$
- **Credit Approval Rate (Favorable Outcome Rate):**
  $$\text{AR}_g = \frac{\text{TN}_g + \text{FN}_g}{N_g} = 1 - \text{SR}_g = P(\hat{Y}=0 \mid A=g)$$

### 2.3 Group Fairness Metrics
Let $u$ denote the **unprivileged group** (Female in Taiwan/German; $\le 25$ in age audit) and $p$ denote the **privileged group** (Male in Taiwan/German; $>25$ in age audit).

1. **Equal Opportunity Difference (EOppD):**
   $$\text{EOppD} = |\text{TPR}_u - \text{TPR}_p|$$
   - *Interpretation:* Measures disparity in correctly identifying default risk across groups. Parity is $0.0$.
2. **Equalized Odds Difference (EOD):**
   $$\text{EOD} = \max\left(|\text{TPR}_u - \text{TPR}_p|,\; |\text{FPR}_u - \text{FPR}_p|\right)$$
   - *Verification:* The code in [`src/modeling/evaluation.py`](file:///d:/sarthak/Subjects/ERA/Project/src/modeling/evaluation.py) uses Fairlearn's `equalized_odds_difference`, which strictly evaluates the supremum of TPR and FPR differences across groups.
3. **Demographic Parity Difference (DPD):**
   $$\text{DPD} = |\text{SR}_u - \text{SR}_p|$$
   - *Interpretation:* Measures the absolute difference in default prediction rates. Parity is $0.0$.
4. **Approval-Rate Disparate Impact (Approval DI):**
   $$\text{Approval DI} = \frac{\text{AR}_u}{\text{AR}_p} = \frac{P(\hat{Y}=0 \mid A=u)}{P(\hat{Y}=0 \mid A=p)}$$
   - *Numerator:* Unprivileged group (Female).
   - *Denominator:* Privileged group (Male).
   - *Parity Benchmark:* Parity is exactly $1.0000$.
   - *Values $< 1.0$:* Unprivileged group receives credit approvals at a lower rate than privileged group.
   - *Values $> 1.0$:* Unprivileged group receives credit approvals at a higher rate.
   - *Distance from Parity:* $|\text{Approval DI} - 1.0|$.

---

## 3. Independent Reconstruction & Recomputation Audit

We executed an independent assertion audit comparing the 18 rows in [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv) against values independently computed from the raw integer confusion matrices in [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv).

### Summary of Independent Recomputation Checks:
- **Total Experiments Audited:** 18
- **Total Subgroup Groups Audited:** 72 (18 experiments $\times$ 2 attributes [sex, age] $\times$ 2 groups)
- **Tolerance Threshold:** $1.0 \times 10^{-4}$
- **Total Calculation Mismatches:** **0 (ZERO)**

### Complete Recomputation Audit Table:
Every performance metric ($\text{Acc}, \text{Prec}, \text{Rec}, \text{F1}$) and fairness metric ($\text{EOppD}, \text{EOD}, \text{DPD}, \text{Approval DI}$) perfectly matches the raw TP, TN, FP, FN counts:

```
[VERIFIED] TAIWAN  | logistic_regression  | reweighing             | EOD: 0.0229 | EOpp: 0.0229 | DPD: 0.0069 | AppDI: 1.0079 | Mismatches: 0
[VERIFIED] TAIWAN  | logistic_regression  | exponentiated_gradient | EOD: 0.0309 | EOpp: 0.0309 | DPD: 0.0047 | AppDI: 1.0053 | Mismatches: 0
[VERIFIED] TAIWAN  | logistic_regression  | threshold_optimizer    | EOD: 0.0425 | EOpp: 0.0425 | DPD: 0.0080 | AppDI: 0.9909 | Mismatches: 0
[VERIFIED] TAIWAN  | random_forest        | reweighing             | EOD: 0.0219 | EOpp: 0.0219 | DPD: 0.0075 | AppDI: 1.0086 | Mismatches: 0
[VERIFIED] TAIWAN  | random_forest        | exponentiated_gradient | EOD: 0.0298 | EOpp: 0.0298 | DPD: 0.0093 | AppDI: 1.0107 | Mismatches: 0
[VERIFIED] TAIWAN  | random_forest        | threshold_optimizer    | EOD: 0.0235 | EOpp: 0.0230 | DPD: 0.0309 | AppDI: 1.0378 | Mismatches: 0
[VERIFIED] TAIWAN  | xgboost              | reweighing             | EOD: 0.0436 | EOpp: 0.0436 | DPD: 0.0003 | AppDI: 1.0003 | Mismatches: 0
[VERIFIED] TAIWAN  | xgboost              | exponentiated_gradient | EOD: 0.0436 | EOpp: 0.0436 | DPD: 0.0007 | AppDI: 0.9992 | Mismatches: 0
[VERIFIED] TAIWAN  | xgboost              | threshold_optimizer    | EOD: 0.0572 | EOpp: 0.0572 | DPD: 0.0096 | AppDI: 0.9891 | Mismatches: 0
[VERIFIED] GERMAN  | logistic_regression  | reweighing             | EOD: 0.1000 | EOpp: 0.1000 | DPD: 0.0131 | AppDI: 1.0177 | Mismatches: 0
[VERIFIED] GERMAN  | logistic_regression  | exponentiated_gradient | EOD: 0.0500 | EOpp: 0.0500 | DPD: 0.0361 | AppDI: 0.9513 | Mismatches: 0
[VERIFIED] GERMAN  | logistic_regression  | threshold_optimizer    | EOD: 0.0500 | EOpp: 0.0500 | DPD: 0.0105 | AppDI: 0.9860 | Mismatches: 0
[VERIFIED] GERMAN  | random_forest        | reweighing             | EOD: 0.0500 | EOpp: 0.0500 | DPD: 0.0005 | AppDI: 0.9994 | Mismatches: 0
[VERIFIED] GERMAN  | random_forest        | exponentiated_gradient | EOD: 0.1250 | EOpp: 0.1250 | DPD: 0.0620 | AppDI: 0.9269 | Mismatches: 0
[VERIFIED] GERMAN  | random_forest        | threshold_optimizer    | EOD: 0.0251 | EOpp: 0.0250 | DPD: 0.0393 | AppDI: 0.9495 | Mismatches: 0
[VERIFIED] GERMAN  | xgboost              | reweighing             | EOD: 0.0250 | EOpp: 0.0250 | DPD: 0.0209 | AppDI: 0.9736 | Mismatches: 0
[VERIFIED] GERMAN  | xgboost              | exponentiated_gradient | EOD: 0.0250 | EOpp: 0.0250 | DPD: 0.0157 | AppDI: 0.9796 | Mismatches: 0
[VERIFIED] GERMAN  | xgboost              | threshold_optimizer    | EOD: 0.0411 | EOpp: 0.0250 | DPD: 0.0373 | AppDI: 0.9529 | Mismatches: 0
```

---

## 4. Data Leakage & Experimental Controls Audit

| Audit Item | Verification Method | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Sensitive Sex Excluded from $X$** | Code inspection of `load_dataset` in `preprocessing.py` | **PASS** | `cols_to_exclude = ["id", "sex", "original_target", "target"]` (Taiwan) and `["personal_status_sex", "original_target", "target"]` (German). |
| **Identifier Excluded from $X$** | Code inspection | **PASS** | `id` in Taiwan excluded from feature matrix. |
| **Preprocessing Fit Isolation** | Inspection of transformer fitting | **PASS** | All encoders/scalers fitted strictly on $X_{\text{train}}$ prior to transformation. |
| **Test Label Isolation** | Code inspection | **PASS** | $y_{\text{test}}$ accessed strictly in final scoring functions; zero usage in mitigation fitting. |
| **Mitigation Fit Isolation** | Code inspection of `mitigation.py` | **PASS** | Sample weights, reductions, and threshold optimizations fitted exclusively on $X_{\text{train}}, y_{\text{train}}, \text{sens\_train}$. |
| **Split Alignment** | Deterministic check | **PASS** | Exact same `test_size=0.20, random_state=42, stratify=y` used across all baseline and mitigated runs. |
| **No Cross-Model Contamination** | Object lifecycle inspection | **PASS** | Models and pipelines re-instantiated fresh inside each loop iteration. |

---

## 5. Audit Findings & Non-ROC Issues

While all discrete metrics (TP, TN, FP, FN, Accuracy, Precision, Recall, F1, EOD, EOpp, DPD, Approval DI) are mathematically perfect and reproducible:
1. **Subgroup Column Naming Inconsistency:** In [`results/baseline_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/baseline_subgroup_metrics.csv), the column is named `selection_rate`, whereas in [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv), it is named `default_prediction_rate`. This should be harmonized to avoid indexing issues in shared analysis tools.
2. **Markdown Summary Table Sync:** As documented in `interpretation_corrections.md`, values in `results/mitigation_summary.md` contained manual transcription discrepancies compared to `results/mitigation_master_results.csv`.
