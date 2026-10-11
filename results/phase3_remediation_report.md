# Phase 3.1 Remediation & Research Baseline Lock Report

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Framework:** FairCredit-AI  
**Phase:** Phase 3.1 — Metric Remediation & Baseline Lock  
**Auditor:** Lead ML Data Engineer & Research Methodology Auditor  
**Date:** October 2026  
**Status:** **PHASE 3 REMEDIATION COMPLETE — READY FOR PHASE 4 DESIGN**

---

## 1. What Was Wrong

During the Phase 3 Research Validation Audit, a critical methodological issue and several reporting discrepancies were identified:
1. **Critical ROC-AUC Reporting Flaw:**
   - In [`src/modeling/train_mitigation.py`](file:///d:/sarthak/Subjects/ERA/Project/src/modeling/train_mitigation.py), binary $\{0, 1\}$ decisions from `ThresholdOptimizer` were cast to float and passed directly to `roc_auc_score(y_true, y_score)`.
   - In `ExponentiatedGradient`, the algorithm converged to a single active binary base estimator ($w_0 = 1.0$) for 4 of 6 configurations (German RF, German XGB, Taiwan RF, Taiwan XGB), outputting degenerate binary predictions $\{0.0, 1.0\}$. In the two Logistic Regression models, it produced a coarse 4-level discrete step function.
   - When binary predictions $\{0, 1\}$ are passed to `roc_auc_score`, the trapezoidal area under the resulting two-segment ROC curve is algebraically identical to **Balanced Accuracy**:
     $$\text{ROC-AUC}_{\text{binary}} = \frac{\text{TPR} + (1 - \text{FPR})}{2} = \frac{\text{TPR} + \text{TNR}}{2} = \text{Balanced Accuracy}$$
   - Comparing this single-operating-point trapezoid against continuous baseline ROC-AUC ($\approx 0.75 - 0.81$) was an invalid apples-to-oranges comparison that created an illusory, artificial drop of $\Delta\text{AUC} \approx -0.07$ to $-0.12$.
2. **Markdown Summary Discrepancies:**
   - The narrative markdown summary ([`results/mitigation_summary.md`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_summary.md)) contained 5 manually transcribed numbers that deviated from the verified master CSV (e.g. reporting German RF Reweighing accuracy as $0.7950$ instead of the verified $0.7800$).
3. **Unsupported Causal Claims:**
   - Literature comments had casually attributed AUC drops to "calibration loss," even though ROC-AUC measures ranking discrimination, not calibration.
4. **Subgroup Column Naming Inconsistency:**
   - Baseline subgroup metrics used `selection_rate`, whereas mitigated subgroup metrics used `default_prediction_rate`.

---

## 2. What Was Corrected

1. **ROC-AUC Qualification & Separation from Balanced Accuracy:**
   - Mitigated ROC-AUC for `ThresholdOptimizer` is now formally designated as **`N/A`** across all tables and CSVs.
   - Mitigated ROC-AUC for `ExponentiatedGradient` is now formally designated as **`N/A`** because it produces a distribution over binary base classifiers rather than a smooth, continuous ranking score.
   - Continuous ROC-AUC is retained **strictly for Reweighing**, which retrains the predictive model and outputs genuine continuous probability estimates $P(Y=1 \mid X)$.
2. **Dedicated Balanced Accuracy Reporting:**
   - Dedicated columns `baseline_balanced_accuracy`, `mitigated_balanced_accuracy`, and `delta_balanced_accuracy` were added to [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv).
   - This provides a direct, mathematically valid comparison between baseline binary decisions and mitigated binary decisions.
3. **Programmatic Summary Regeneration:**
   - [`results/mitigation_summary.md`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_summary.md) was regenerated programmatically via [`scripts/generate_mitigation_summary.py`](file:///d:/sarthak/Subjects/ERA/Project/scripts/generate_mitigation_summary.py) directly from [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv), eliminating 100% of narrative transcription discrepancies.
4. **Harmonized Subgroup Naming:**
   - Added `selection_rate` to [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv) to achieve complete schema alignment with [`results/baseline_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/baseline_subgroup_metrics.csv).
5. **Causal Claims Removed:**
   - All speculative statements attributing performance changes to "calibration loss" were replaced with scientifically defensible wording detailing the discrete predictor mechanics.
   - Over-generalized claims were replaced with configuration-specific empirical facts.

---

## 3. Which Metrics Remain Valid

All underlying experimental executions and mathematical metrics remain **100% valid and verified**:
- **Discrete Classification Performance:** Accuracy, Precision, Recall, F1-Score.
- **Subgroup Confusion Matrix Counts:** TP, TN, FP, FN, TPR, FPR, selection rate, approval rate across all 72 subgroups.
- **Fairness Metrics (Exact Formulations):**
  - **Equalized Odds Difference:** $\max(|\text{TPR}_u - \text{TPR}_p|, |\text{FPR}_u - \text{FPR}_p|)$
  - **Equal Opportunity Difference:** $|\text{TPR}_u - \text{TPR}_p|$
  - **Demographic Parity Difference:** $|\text{SelectionRate}_u - \text{SelectionRate}_p|$
  - **Approval Disparate Impact:** $\text{ApprovalRate}_u / \text{ApprovalRate}_p$
  - **Distance from Parity:** $|\text{Approval DI} - 1.0|$
- **Target Convention:** $Y=1$ Default / Bad Risk, $Y=0$ Non-Default / Favorable Credit Approval.

---

## 4. Which ROC-AUC Values Are Now N/A

| Mitigation Method | Model Type | Mitigated ROC-AUC Status | Rationale |
| :--- | :--- | :---: | :--- |
| **ThresholdOptimizer** | Post-Processing Decision Rule | **`N/A`** | Operates strictly at the decision threshold level on prefit probabilities; outputs discrete classifications $\hat{Y} \in \{0, 1\}$, not continuous ranking scores. |
| **Exponentiated Gradient** | In-Processing Reduction (`EqualizedOdds`) | **`N/A`** | Returns a randomized mixture of binary base classifiers. Output distribution is discrete/step-wise ($\le 4$ discrete values or binary), lacking a fine-grained continuous ranking function. |
| **Reweighing** | Pre-Processing Sample Weighting | **RETAINED (Continuous)** | Retrains the base classifier under adjusted sample weights, yielding a newly fitted continuous scoring function $P(Y=1 \mid X)$. |

---

## 5. Where Balanced Accuracy Is Reported

Balanced Accuracy is now reported in dedicated columns in:
1. [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv):
   - `baseline_balanced_accuracy`
   - `mitigated_balanced_accuracy`
   - `delta_balanced_accuracy`
2. [`results/mitigation_summary.md`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_summary.md):
   - Displayed prominently in Master Tables 2.1 and 2.2 (`Base BalAcc`, `Mit BalAcc`, `Δ BalAcc`).
3. [`scripts/verify_mitigation_results.py`](file:///d:/sarthak/Subjects/ERA/Project/scripts/verify_mitigation_results.py):
   - Fully integrated into the automated assertion test suite.

**Key Insight Revealed by Balanced Accuracy:**  
When evaluated on Balanced Accuracy, `ThresholdOptimizer` actually **improved** balanced accuracy in 5 of 6 models (e.g. German RF: $0.6762 \to 0.6774$, Taiwan RF: $0.6499 \to 0.6821$) because threshold optimization increased recall on the minority default class. The previous apparent $-10\%$ drop was purely an artifact of comparing binary Balanced Accuracy against continuous AUC.

---

## 6. Which Methods Provide Continuous Scoring Outputs

- **Baseline Models (Logistic Regression, Random Forest, XGBoost):** **YES** (Standard continuous `predict_proba`).
- **Reweighing (Pre-Processing):** **YES** (Standard continuous `predict_proba` fitted under statistical independence sample weights).
- **Exponentiated Gradient (In-Processing):** **NO** (Outputs randomized binary decisions; ensemble PMF is discrete/step-wise).
- **ThresholdOptimizer (Post-Processing):** **NO** (Outputs discrete binary decisions $\hat{Y} \in \{0, 1\}$).

---

## 7. Which Methods Are Eligible for Scoring-Model SHAP Analysis

| Mitigation Method | Eligible for Scoring-Model SHAP? | Technical Rationale |
| :--- | :---: | :--- |
| **Reweighing** | **YES (Primary Target)** | Directly modifies the underlying model weights/trees through weighted empirical risk minimization. Produces a smooth, continuous scoring model $f_{\text{reweighed}}(x)$. Fully compatible with TreeSHAP and KernelSHAP. |
| **Exponentiated Gradient** | **CONDITIONAL / SPECIAL PROTOCOL** | Forms an ensemble of binary cost-sensitive estimators. TreeSHAP/LinearSHAP cannot be applied naively to the top-level reduction object. Requires either explaining individual active base estimators or computing KernelSHAP over the weighted ensemble step function. |
| **ThresholdOptimizer** | **NO (Excluded from Model SHAP)** | Post-processes the final binary classification cutoff. The underlying model scoring function $f(x)$ is **completely unmodified**. Computing SHAP on the underlying model would produce explanations identically equal to the baseline. |

---

## 8. Which Method Is Decision-Level Only

- **`ThresholdOptimizer`** is strictly a **decision-level, post-processing intervention**.
- It is retained in FairCredit-AI for:
  - Decision-level fairness auditing
  - Accuracy vs. error-rate parity trade-off analysis
  - Comparative benchmarking against in-processing and pre-processing interventions

---

## 9. Updated Authoritative Result Tables

### 9.1 Taiwan Credit Card Default ($N_{\text{test}} = 6,000$, Primary: Sex)

| Model | Mitigation Method | Base Acc | Mit Acc | Δ Acc | Base BalAcc | Mit BalAcc | Δ BalAcc | Base AUC | Mit AUC | Δ AUC | Base EOD | Mit EOD | Δ EOD | Base DPD | Mit DPD | Δ DPD | Base App-DI | Mit App-DI | Base Dist | Mit Dist | Closer to Parity? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Reweighing** | 0.8173 | 0.8175 | +0.0002 | 0.6512 | 0.6508 | -0.0004 | 0.7574 | 0.7582 | +0.0008 | 0.0194 | 0.0229 | +0.0035 | 0.0069 | 0.0069 | +0.0000 | 1.0079 | 1.0079 | 0.0079 | 0.0079 | Stable |
| **Logistic Regression** | **Exp. Gradient** | 0.8173 | 0.8183 | +0.0010 | 0.6512 | 0.6511 | -0.0001 | 0.7574 | **N/A** | **N/A** | 0.0194 | 0.0309 | +0.0115 | 0.0069 | 0.0047 | -0.0022 | 1.0079 | 1.0053 | 0.0079 | 0.0053 | **Yes** |
| **Logistic Regression** | **ThresholdOpt** | 0.8173 | 0.8143 | -0.0030 | 0.6512 | 0.6598 | +0.0086 | 0.7574 | **N/A** | **N/A** | 0.0194 | 0.0425 | +0.0231 | 0.0069 | 0.0080 | +0.0011 | 1.0079 | 0.9909 | 0.0079 | 0.0091 | No |
| **Random Forest** | **Reweighing** | 0.8118 | 0.8123 | +0.0005 | 0.6499 | 0.6505 | +0.0006 | 0.7528 | 0.7548 | +0.0020 | 0.0298 | 0.0219 | -0.0079 | 0.0093 | 0.0075 | -0.0018 | 1.0107 | 1.0086 | 0.0107 | 0.0086 | **Yes** |
| **Random Forest** | **Exp. Gradient** | 0.8118 | 0.8118 | +0.0000 | 0.6499 | 0.6499 | +0.0000 | 0.7528 | **N/A** | **N/A** | 0.0298 | 0.0298 | +0.0000 | 0.0093 | 0.0093 | +0.0000 | 1.0107 | 1.0107 | 0.0107 | 0.0107 | Stable |
| **Random Forest** | **ThresholdOpt** | 0.8118 | 0.8125 | +0.0007 | 0.6499 | 0.6821 | +0.0322 | 0.7528 | **N/A** | **N/A** | 0.0298 | 0.0235 | -0.0063 | 0.0093 | 0.0309 | +0.0216 | 1.0107 | 1.0378 | 0.0107 | 0.0378 | No |
| **XGBoost** | **Reweighing** | 0.8193 | 0.8190 | -0.0003 | 0.6555 | 0.6529 | -0.0026 | 0.7785 | 0.7774 | -0.0011 | 0.0306 | 0.0436 | +0.0130 | 0.0027 | 0.0003 | -0.0024 | 1.0031 | 1.0003 | 0.0031 | 0.0003 | **Yes** |
| **XGBoost** | **Exp. Gradient** | 0.8193 | 0.8188 | -0.0005 | 0.6555 | 0.6528 | -0.0027 | 0.7785 | **N/A** | **N/A** | 0.0306 | 0.0436 | +0.0130 | 0.0027 | 0.0007 | -0.0020 | 1.0031 | 0.9992 | 0.0031 | 0.0008 | **Yes** |
| **XGBoost** | **ThresholdOpt** | 0.8193 | 0.8183 | -0.0010 | 0.6555 | 0.6597 | +0.0042 | 0.7785 | **N/A** | **N/A** | 0.0306 | 0.0572 | +0.0266 | 0.0027 | 0.0096 | +0.0069 | 1.0031 | 0.9891 | 0.0031 | 0.0109 | No |

---

### 9.2 Statlog German Credit ($N_{\text{test}} = 200$, Primary: Derived Sex)

| Model | Mitigation Method | Base Acc | Mit Acc | Δ Acc | Base BalAcc | Mit BalAcc | Δ BalAcc | Base AUC | Mit AUC | Δ AUC | Base EOD | Mit EOD | Δ EOD | Base DPD | Mit DPD | Δ DPD | Base App-DI | Mit App-DI | Base Dist | Mit Dist | Closer to Parity? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Reweighing** | 0.7850 | 0.7850 | +0.0000 | 0.7226 | 0.7226 | +0.0000 | 0.8061 | 0.8055 | -0.0006 | 0.0500 | 0.1000 | +0.0500 | 0.0577 | 0.0131 | -0.0446 | 0.9244 | 1.0177 | 0.0756 | 0.0177 | **Yes** |
| **Logistic Regression** | **Exp. Gradient** | 0.7850 | 0.7800 | -0.0050 | 0.7226 | 0.7238 | +0.0012 | 0.8061 | **N/A** | **N/A** | 0.0500 | 0.0500 | +0.0000 | 0.0577 | 0.0361 | -0.0216 | 0.9244 | 0.9513 | 0.0756 | 0.0487 | **Yes** |
| **Logistic Regression** | **ThresholdOpt** | 0.7850 | 0.7650 | -0.0200 | 0.7226 | 0.6988 | -0.0238 | 0.8061 | **N/A** | **N/A** | 0.0500 | 0.0500 | +0.0000 | 0.0577 | 0.0105 | -0.0472 | 0.9244 | 0.9860 | 0.0756 | 0.0140 | **Yes** |
| **Random Forest** | **Reweighing** | 0.7800 | 0.7800 | +0.0000 | 0.6762 | 0.6810 | +0.0048 | 0.7798 | 0.7935 | +0.0137 | 0.1250 | 0.0500 | -0.0750 | 0.0620 | 0.0005 | -0.0615 | 0.9269 | 0.9994 | 0.0731 | 0.0006 | **Yes** |
| **Random Forest** | **Exp. Gradient** | 0.7800 | 0.7800 | +0.0000 | 0.6762 | 0.6762 | +0.0000 | 0.7798 | **N/A** | **N/A** | 0.1250 | 0.1250 | +0.0000 | 0.0620 | 0.0620 | +0.0000 | 0.9269 | 0.9269 | 0.0731 | 0.0731 | Stable |
| **Random Forest** | **ThresholdOpt** | 0.7800 | 0.7550 | -0.0250 | 0.6762 | 0.6774 | +0.0012 | 0.7798 | **N/A** | **N/A** | 0.1250 | 0.0251 | -0.0999 | 0.0620 | 0.0393 | -0.0227 | 0.9269 | 0.9495 | 0.0731 | 0.0505 | **Yes** |
| **XGBoost** | **Reweighing** | 0.7750 | 0.7750 | +0.0000 | 0.7012 | 0.6917 | -0.0095 | 0.8019 | 0.7918 | -0.0101 | 0.0250 | 0.0250 | +0.0000 | 0.0157 | 0.0209 | +0.0052 | 0.9796 | 0.9736 | 0.0204 | 0.0264 | No |
| **XGBoost** | **Exp. Gradient** | 0.7750 | 0.7750 | +0.0000 | 0.7012 | 0.7012 | +0.0000 | 0.8019 | **N/A** | **N/A** | 0.0250 | 0.0250 | +0.0000 | 0.0157 | 0.0157 | +0.0000 | 0.9796 | 0.9796 | 0.0204 | 0.0204 | Stable |
| **XGBoost** | **ThresholdOpt** | 0.7750 | 0.7900 | +0.0150 | 0.7012 | 0.7119 | +0.0107 | 0.8019 | **N/A** | **N/A** | 0.0250 | 0.0411 | +0.0161 | 0.0157 | 0.0373 | +0.0216 | 0.9796 | 0.9529 | 0.0204 | 0.0471 | No |

---

## 10. Confirmation of Zero Silent Changes

- **Underlying Experimental Integrity:** No models were retrained, no datasets were modified, no hyperparameters were altered, and no raw numbers were overwritten.
- **Deltas:** Every single delta ($\Delta\text{Acc}, \Delta\text{BalAcc}, \Delta\text{F1}, \Delta\text{EOD}, \Delta\text{DPD}$) strictly matches subtraction between baseline and mitigated metrics.
- **Assertion Verification:** Running [`scripts/verify_mitigation_results.py`](file:///d:/sarthak/Subjects/ERA/Project/scripts/verify_mitigation_results.py) passes 100% of all mathematical assertions with **0 errors and 0 mismatches**.

---

## 11. Confirmation of Zero Data Leakage

- **Feature Matrix Isolation:** Protected attribute `sex` (Taiwan) and `personal_status_sex` / `derived_sex` (German), as well as identifiers (`id`), remain **strictly excluded from feature matrix $X$**.
- **Preprocessing Isolation:** All encoders and scalers were fitted strictly on $X_{\text{train}}$.
- **Mitigation Fitting:** Weights, reduction estimators, and group thresholds were fitted strictly on $(X_{\text{train}}, y_{\text{train}}, \text{sens\_train})$.
- **Evaluation Isolation:** Ground-truth test labels $y_{\text{test}}$ were accessed solely for final performance scoring. Zero test leakage occurred.

---

## 12. Final Recommendation for Phase 4 (SHAP & Explanation Consistency)

1. **Primary Experimental Axis:** Focus the main SHAP explanation-consistency analysis on **Baseline vs. Reweighing**:
   - Both models possess continuous scoring functions $P(Y=1 \mid X)$.
   - Reweighing alters the underlying model representations and decision boundaries via sample weighting.
   - Explanations (TreeSHAP for Random Forest & XGBoost, LinearSHAP for Logistic Regression) are well-defined, continuous, and directly comparable.
2. **Secondary Exploration (Exponentiated Gradient):** If Exponentiated Gradient is analyzed under SHAP, explicitly explain the dominant base estimator ($w_0 = 1.0$) or compute a KernelSHAP approximation over the weighted mixture, noting its randomized discrete structure.
3. **ThresholdOptimizer Protocol:** Retain ThresholdOptimizer exclusively for decision-level trade-off analysis; do not claim it alters local feature explanations.
