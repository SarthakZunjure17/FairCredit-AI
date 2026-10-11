# Methodological & Implementation Fixes Required

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Document:** Required Fixes Prior to Phase 4 (SHAP & Explanation Consistency)  
**Auditor:** Lead ML Data Engineer & Research Methodology Auditor  
**Date:** October 2026  
**Status:** **ACTIONABLE REMEDIATION PLAN DOCUMENTED**

---

## 1. Executive Summary

This document specifies the required code, schema, and reporting modifications identified during the Phase 3 methodological audit.

Per user instructions:
> *"Do NOT automatically rewrite the entire Phase 3 pipeline. Your job is to determine whether our existing Phase 3 results are scientifically valid and whether the metrics are being calculated correctly."*

No destructive changes or unilateral pipeline rewrites have been executed. The fixes below are formally cataloged for user review and approval before proceeding to Phase 4.

---

## 2. Priority 1 (CRITICAL): ROC-AUC Reporting & Metric Definition Fix

### 2.1 The Issue
- In [`src/modeling/train_mitigation.py`](file:///d:/sarthak/Subjects/ERA/Project/src/modeling/train_mitigation.py), line 200:
  `y_prob = y_pred.astype(float)` passes binary $\{0, 1\}$ decisions from `ThresholdOptimizer` into `roc_auc_score`.
- In line 187, `pmf = model_obj._pmf_predict(X_test_proc)` passes degenerate binary vectors $\{0, 1\}$ into `roc_auc_score` for 4 of 6 `ExponentiatedGradient` models (German RF, German XGB, Taiwan RF, Taiwan XGB).
- `roc_auc_score` on binary predictions calculates **Balanced Accuracy**, not continuous ROC-AUC.
- Comparing binary Balanced Accuracy ($\approx 0.65 - 0.71$) against continuous baseline ROC-AUC ($\approx 0.75 - 0.81$) generates an artificial, scientifically invalid performance drop of $\Delta\text{AUC} \approx -0.07$ to $-0.12$.

### 2.2 Proposed Remediation Options
To make Phase 3 publication-ready and mathematically defensible, select one of the following two standard academic approaches:

- **Option A (Recommended: Dedicated Balanced Accuracy Column & Explicit ROC-AUC Qualification):**
  1. Add explicit `baseline_balanced_accuracy`, `mitigated_balanced_accuracy`, and `delta_balanced_accuracy` columns to `results/mitigation_master_results.csv`.
     - *Benefit:* Both baseline and mitigated models are evaluated on identical binary decision operating points. Baseline Balanced Accuracy can be directly compared with ThresholdOptimizer Balanced Accuracy (e.g. German RF baseline BalAcc = 0.6773 vs ThresholdOptimizer BalAcc = 0.6774, showing zero penalty rather than a 10% drop!).
  2. For `ThresholdOptimizer`, set `mitigated_roc_auc` to `N/A (Decision Rule)` or record the underlying baseline model AUC, explicitly noting that post-processing does not alter the underlying model's continuous risk score.
  3. For `ExponentiatedGradient`, clearly document in footnote and metadata that reported AUC is a discrete step-approximation derived from a finite ensemble of binary base classifiers.

- **Option B (Separate Decision-Level Benchmark Table):**
  - Partition `results/mitigation_master_results.csv` into two sub-tables:
    - *Table 1 (Representation / Scoring Mitigations):* Reweighing vs. Baseline (Continuous ROC-AUC, F1, Accuracy, EOD, DPD).
    - *Table 2 (Decision-Level Interventions):* ThresholdOptimizer & Exponentiated Gradient vs. Baseline (Accuracy, Balanced Accuracy, Precision, Recall, EOD, DPD, Approval DI).

---

## 3. Priority 2: Synchronize Markdown Summary Table with Verified CSV

### 3.1 The Issue
[`results/mitigation_summary.md`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_summary.md) contains manually transcribed numbers in Tables 2.1 and 2.2 that deviate from the verified values in [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv).

Specific deviations:
1. **German RF Reweighing Accuracy:** Summary claimed $0.7950$ (+0.0150). Verified CSV is **0.7800** (+0.0000).
2. **German LR Reweighing Accuracy:** Summary claimed $0.7900$ (+0.0050). Verified CSV is **0.7850** (+0.0000).
3. **German LR ThresholdOpt Accuracy:** Summary claimed $0.7350$ (-0.0500). Verified CSV is **0.7650** (-0.0200).
4. **German RF ThresholdOpt Accuracy:** Summary claimed $0.7100$ (-0.0700). Verified CSV is **0.7550** (-0.0250).
5. **German XGB ThresholdOpt Accuracy:** Summary claimed $0.7250$ (-0.0500). Verified CSV is **0.7900** (+0.0150).

### 3.2 Required Remediation
- Re-generate Markdown Tables 2.1 and 2.2 in `results/mitigation_summary.md` directly from `results/mitigation_master_results.csv` via automated script to guarantee 100% synchronization.
- Update narrative in Section 3 of `mitigation_summary.md` to reflect the exact verified figures.

---

## 4. Priority 3: Terminology and Conceptual Refinements

### 4.1 Exponentiated Gradient Optimization Goal
- In all reports and documentation, state clearly that Exponentiated Gradient was configured with the **`EqualizedOdds()`** constraint.
- Remove statements implying that it was optimized for Demographic Parity. Report Demographic Parity changes strictly as secondary observed empirical effects.

### 4.2 Causal Statements
- Remove all references attributing AUC reduction to "calibration loss."
- Replace with scientific wording explaining the mathematical identity between binary prediction AUC and Balanced Accuracy.

---

## 5. Priority 4: Subgroup Column Harmonization

### 5.1 The Issue
- In [`results/baseline_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/baseline_subgroup_metrics.csv), the rate of predicting default is named: `selection_rate`.
- In [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv), it is named: `default_prediction_rate`.

### 5.2 Required Remediation
- Include both `selection_rate` and `default_prediction_rate` (or standardize to `selection_rate (default_rate)`) in both CSVs to prevent key errors in shared plotting and audit scripts.

---

## 6. Priority 5: Protocol for Phase 4 (SHAP & Explanation Consistency)

### 6.1 ThresholdOptimizer Exclusion from Scoring-Model SHAP
- ThresholdOptimizer must **NOT** be included in SHAP analysis of the continuous predictive scoring model.
- Because ThresholdOptimizer post-processes final decisions without modifying the baseline model's scoring function, computing TreeSHAP or KernelSHAP on the underlying model would produce explanations identical to the unmitigated baseline.
- **Phase 4 Target Models:** Explanation consistency analysis before and after fairness mitigation must focus on models where the underlying hypothesis / predictive function was modified:
  1. **Baseline Model** vs. **Reweighted Model (Pre-processing)**
  2. **Baseline Model** vs. **Exponentiated Gradient Base Estimators (In-processing)**
