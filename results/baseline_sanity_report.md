# Baseline Model Sanity & Metric Reconstruction Report

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Audit Phase:** Pre-Mitigation Baseline Sanity Review  
**Auditor:** Lead ML Data Engineer  
**Date:** October 2026  
**Status:** ALL CHECKS PASSED — MATHEMATICALLY VERIFIED  

---

## 1. Executive Summary

Prior to introducing fairness mitigation techniques (Fairlearn re-weighting, grid search, exponentiated gradient, post-processing thresholding) or SHAP explanations, a strict **Baseline Sanity Review** was conducted to verify that:
1. All baseline metrics in `results/baseline_master_results.csv` exactly match empirical predictions.
2. Fairness metrics follow documented mathematical definitions and can be independently reconstructed from raw subgroup confusion matrices.
3. Equalized Odds Difference strictly computes $\max(|\Delta\text{TPR}|, |\Delta\text{FPR}|)$ across groups.
4. Primary protected attributes (`sex` in Taiwan, `personal_status_sex` / `derived_sex` in German) and customer identifiers (`id` in Taiwan) were **never included in feature matrix $X$**.
5. Preprocessing (One-Hot Encoding, StandardScaler) was fitted **strictly on training data** with zero leakage from test splits or targets.

**Audit Verdict:** $\mathbf{PASS\ (100\%\ Verified)}$.

---

## 2. Independent Subgroup Confusion Matrices

*Generated and saved to [`results/baseline_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/baseline_subgroup_metrics.csv)*

### 2.1 Taiwan Credit Card Default ($N_{\text{test}} = 6,000$, Features $P=22$)

#### Primary Protected Attribute: `sex` (Female: $N=3,598$; Male: $N=2,402$)
| Model | Group | Group Role | $N$ | TP | TN | FP | FN | TPR | FPR | Selection Rate ($P(\hat{Y}=1)$) | Approval Rate ($P(\hat{Y}=0)$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Female | Unprivileged | 3,598 | 277 | 2,695 | 137 | 489 | **0.3616** | **0.0484** | **0.1151** | **0.8849** |
| **Logistic Regression** | Male | Privileged | 2,402 | 192 | 1,740 | 101 | 369 | **0.3422** | **0.0549** | **0.1220** | **0.8780** |
| **Random Forest** | Female | Unprivileged | 3,598 | 285 | 2,677 | 155 | 481 | **0.3721** | **0.0547** | **0.1223** | **0.8777** |
| **Random Forest** | Male | Privileged | 2,402 | 192 | 1,717 | 124 | 369 | **0.3422** | **0.0674** | **0.1316** | **0.8684** |
| **XGBoost** | Female | Unprivileged | 3,598 | 287 | 2,693 | 139 | 479 | **0.3747** | **0.0491** | **0.1184** | **0.8816** |
| **XGBoost** | Male | Privileged | 2,402 | 193 | 1,743 | 98 | 368 | **0.3440** | **0.0532** | **0.1211** | **0.8789** |

#### Secondary Protected Attribute: `age` ($\le 25$: $N=767$; $>25$: $N=5,233$)
| Model | Group | Group Role | $N$ | TP | TN | FP | FN | TPR | FPR | Selection Rate ($P(\hat{Y}=1)$) | Approval Rate ($P(\hat{Y}=0)$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $\le 25$ | Unprivileged | 767 | 87 | 524 | 47 | 109 | **0.4439** | **0.0823** | **0.1747** | **0.8253** |
| **Logistic Regression** | $>25$ | Privileged | 5,233 | 382 | 3,911 | 191 | 749 | **0.3378** | **0.0466** | **0.1095** | **0.8905** |
| **Random Forest** | $\le 25$ | Unprivileged | 767 | 94 | 520 | 51 | 102 | **0.4796** | **0.0893** | **0.1890** | **0.8110** |
| **Random Forest** | $>25$ | Privileged | 5,233 | 383 | 3,874 | 228 | 748 | **0.3386** | **0.0556** | **0.1168** | **0.8832** |
| **XGBoost** | $\le 25$ | Unprivileged | 767 | 94 | 522 | 49 | 102 | **0.4796** | **0.0858** | **0.1864** | **0.8136** |
| **XGBoost** | $>25$ | Privileged | 5,233 | 386 | 3,914 | 188 | 745 | **0.3413** | **0.0458** | **0.1097** | **0.8903** |

---

### 2.2 Statlog German Credit ($N_{\text{test}} = 200$, Features $P=19$)

#### Primary Protected Attribute: `sex` (Female: $N=61$; Male: $N=139$)
| Model | Group | Group Role | $N$ | TP | TN | FP | FN | TPR | FPR | Selection Rate ($P(\hat{Y}=1)$) | Approval Rate ($P(\hat{Y}=0)$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Female | Unprivileged | 61 | 12 | 35 | 6 | 8 | **0.6000** | **0.1463** | **0.2951** | **0.7049** |
| **Logistic Regression** | Male | Privileged | 139 | 22 | 88 | 11 | 18 | **0.5500** | **0.1111** | **0.2374** | **0.7626** |
| **Random Forest** | Female | Unprivileged | 61 | 10 | 38 | 3 | 10 | **0.5000** | **0.0732** | **0.2131** | **0.7869** |
| **Random Forest** | Male | Privileged | 139 | 15 | 93 | 6 | 25 | **0.3750** | **0.0606** | **0.1511** | **0.8489** |
| **XGBoost** | Female | Unprivileged | 61 | 10 | 36 | 5 | 10 | **0.5000** | **0.1220** | **0.2459** | **0.7541** |
| **XGBoost** | Male | Privileged | 139 | 21 | 88 | 11 | 19 | **0.5250** | **0.1111** | **0.2302** | **0.7698** |

#### Secondary Protected Attribute: `age` ($\le 25$: $N=44$; $>25$: $N=156$)
| Model | Group | Group Role | $N$ | TP | TN | FP | FN | TPR | FPR | Selection Rate ($P(\hat{Y}=1)$) | Approval Rate ($P(\hat{Y}=0)$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $\le 25$ | Unprivileged | 44 | 11 | 19 | 5 | 9 | **0.5500** | **0.2083** | **0.3636** | **0.6364** |
| **Logistic Regression** | $>25$ | Privileged | 156 | 23 | 104 | 12 | 17 | **0.5750** | **0.1034** | **0.2244** | **0.7756** |
| **Random Forest** | $\le 25$ | Unprivileged | 44 | 7 | 23 | 1 | 13 | **0.3500** | **0.0417** | **0.1818** | **0.8182** |
| **Random Forest** | $>25$ | Privileged | 156 | 18 | 108 | 8 | 22 | **0.4500** | **0.0690** | **0.1667** | **0.8333** |
| **XGBoost** | $\le 25$ | Unprivileged | 44 | 9 | 21 | 3 | 11 | **0.4500** | **0.1250** | **0.2727** | **0.7273** |
| **XGBoost** | $>25$ | Privileged | 156 | 22 | 103 | 13 | 18 | **0.5500** | **0.1121** | **0.2244** | **0.7756** |

---

## 3. Independent Reconstruction & Verification of Fairness Metrics

Every metric reported in `results/baseline_master_results.csv` and `results/baseline_fairness.csv` was recalculated directly from the confusion matrix values above.

### 3.1 Equalized Odds Difference Verification ($\max(|\Delta\text{TPR}|, |\Delta\text{FPR}|)$)
We verify that Equalized Odds is strictly the maximum disparity across both True Positive Rate and False Positive Rate:

1. **Taiwan — Logistic Regression (Sex):**
   - $|\Delta\text{TPR}| = |0.3616 - 0.3422| = \mathbf{0.0194}$
   - $|\Delta\text{FPR}| = |0.0484 - 0.0549| = 0.0065$
   - $\text{Equalized Odds Diff} = \max(0.0194, 0.0065) = \mathbf{0.0194}$ $\checkmark$ *(Matches master result: 0.0194)*

2. **Taiwan — Random Forest (Sex):**
   - $|\Delta\text{TPR}| = |0.3721 - 0.3422| = \mathbf{0.0298}$
   - $|\Delta\text{FPR}| = |0.0547 - 0.0674| = 0.0127$
   - $\text{Equalized Odds Diff} = \max(0.0298, 0.0127) = \mathbf{0.0298}$ $\checkmark$ *(Matches master result: 0.0298)*

3. **Taiwan — XGBoost (Sex):**
   - $|\Delta\text{TPR}| = |0.3747 - 0.3440| = \mathbf{0.0306}$
   - $|\Delta\text{FPR}| = |0.0491 - 0.0532| = 0.0041$
   - $\text{Equalized Odds Diff} = \max(0.0306, 0.0041) = \mathbf{0.0306}$ $\checkmark$ *(Matches master result: 0.0306)*

4. **German — Logistic Regression (Sex):**
   - $|\Delta\text{TPR}| = |0.6000 - 0.5500| = \mathbf{0.0500}$
   - $|\Delta\text{FPR}| = |0.1463 - 0.1111| = 0.0352$
   - $\text{Equalized Odds Diff} = \max(0.0500, 0.0352) = \mathbf{0.0500}$ $\checkmark$ *(Matches master result: 0.0500)*

5. **German — Random Forest (Sex):**
   - $|\Delta\text{TPR}| = |0.5000 - 0.3750| = \mathbf{0.1250}$
   - $|\Delta\text{FPR}| = |0.0732 - 0.0606| = 0.0126$
   - $\text{Equalized Odds Diff} = \max(0.1250, 0.0126) = \mathbf{0.1250}$ $\checkmark$ *(Matches master result: 0.1250)*

6. **German — XGBoost (Sex):**
   - $|\Delta\text{TPR}| = |0.5000 - 0.5250| = \mathbf{0.0250}$
   - $|\Delta\text{FPR}| = |0.1220 - 0.1111| = 0.0109$
   - $\text{Equalized Odds Diff} = \max(0.0250, 0.0109) = \mathbf{0.0250}$ $\checkmark$ *(Matches master result: 0.0250)*

7. **German — Logistic Regression (Age $\le 25$ vs $>25$) [Crucial Case where $\Delta\text{FPR} > \Delta\text{TPR}$]:**
   - $|\Delta\text{TPR}| = |0.5500 - 0.5750| = 0.0250$
   - $|\Delta\text{FPR}| = |0.2083 - 0.1034| = \mathbf{0.1049}$
   - $\text{Equalized Odds Diff} = \max(0.0250, 0.1049) = \mathbf{0.1049}$ $\checkmark$
   - *This explicitly proves that Equalized Odds properly selects the maximum over both TPR and FPR.*

---

### 3.2 Demographic Parity Difference Verification ($|\text{SR}_u - \text{SR}_p|$)
- **Taiwan LR:** $|0.1151 - 0.1220| = \mathbf{0.0069}$ $\checkmark$ *(Matches master: 0.0069)*
- **Taiwan RF:** $|0.1223 - 0.1316| = \mathbf{0.0093}$ $\checkmark$ *(Matches master: 0.0093)*
- **Taiwan XGB:** $|0.1184 - 0.1211| = \mathbf{0.0027}$ $\checkmark$ *(Matches master: 0.0027)*
- **German LR:** $|0.2951 - 0.2374| = \mathbf{0.0577}$ $\checkmark$ *(Matches master: 0.0577)*
- **German RF:** $|0.2131 - 0.1511| = \mathbf{0.0620}$ $\checkmark$ *(Matches master: 0.0620)*
- **German XGB:** $|0.2459 - 0.2302| = \mathbf{0.0157}$ $\checkmark$ *(Matches master: 0.0157)*

---

### 3.3 Disparate Impact (Selection Ratio vs. Approval Ratio)
- **Selection Ratio (Fairlearn convention):** $\frac{\min(\text{SR}_u, \text{SR}_p)}{\max(\text{SR}_u, \text{SR}_p)}$
  - Taiwan LR: $0.1151 / 0.1220 = \mathbf{0.9433}$ $\checkmark$
  - Taiwan RF: $0.1223 / 0.1316 = \mathbf{0.9296}$ $\checkmark$
  - Taiwan XGB: $0.1184 / 0.1211 = \mathbf{0.9773}$ $\checkmark$
  - German LR: $0.2374 / 0.2951 = \mathbf{0.8046}$ $\checkmark$
  - German RF: $0.1511 / 0.2131 = \mathbf{0.7089}$ $\checkmark$
  - German XGB: $0.2302 / 0.2459 = \mathbf{0.9362}$ $\checkmark$
- **Approval Ratio (Banking 80% Rule convention):** $\frac{\text{ApprovalRate}_{\text{female}}}{\text{ApprovalRate}_{\text{male}}}$
  - Taiwan LR: $0.8849 / 0.8780 = \mathbf{1.0079}$
  - Taiwan RF: $0.8777 / 0.8684 = \mathbf{1.0107}$
  - Taiwan XGB: $0.8816 / 0.8789 = \mathbf{1.0031}$
  - German LR: $0.7049 / 0.7626 = \mathbf{0.9244}$
  - German RF: $0.7869 / 0.8489 = \mathbf{0.9269}$
  - German XGB: $0.7541 / 0.7698 = \mathbf{0.9796}$

---

## 4. Leakage & Feature Separation Verification

We verified the code in `src/modeling/preprocessing.py`, `src/modeling/models.py`, and `src/modeling/train.py` against all leakage vectors:

### 4.1 Sensitive Attribute Isolation Check
- **Taiwan:**
  - `sex` was stripped from $X$ via `cols_to_exclude = ["id", "sex", "original_target", "target"]`.
  - Feature matrix $X_{\text{train}}$ contains exactly 22 columns.
  - Confirmed: `sex` is present only in `sensitive_df` passed to evaluation.
- **German:**
  - `personal_status_sex` (the original combined gender/marital status attribute) was stripped from $X$ via `cols_to_exclude = ["personal_status_sex", "original_target", "target"]`.
  - Feature matrix $X_{\text{train}}$ contains exactly 19 columns.
  - Confirmed: `derived_sex` was derived from `personal_status_sex` and retained exclusively in `sensitive_df`. Neither `personal_status_sex` nor `derived_sex` exists in $X$.

### 4.2 Customer Identifier Isolation Check
- `id` in Taiwan exists exclusively in the canonical dataframe and was explicitly excluded from $X$.
- Verified: `assert "id" not in X.columns` $\implies$ **PASSED**.

### 4.3 Target Isolation Check
- `target` and `original_target` were verified absent from $X$ across both datasets.
- Target series $y$ is isolated throughout all pipeline training calls.

### 4.4 Training-Only Preprocessing Verification
- Preprocessors (`ColumnTransformer`) were assembled inside `sklearn.pipeline.Pipeline`.
- Execution calls: `pipeline.fit(X_train, y_train)` fits OneHotEncoder and StandardScaler strictly on the 80% training slice.
- `pipeline.predict(X_test)` and `pipeline.predict_proba(X_test)` apply pre-fitted parameters to test inputs without refitting or computing test-fold statistics.

---

## 5. Artifact Verification & Deliverables

The sanity review generated the required machine-readable subgroup audit table:
- [`results/baseline_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/baseline_subgroup_metrics.csv)
- [`results/baseline_sanity_report.md`](file:///d:/sarthak/Subjects/ERA/Project/results/baseline_sanity_report.md)

### Verification Summary Checklist:
- [x] All 6 rows in `results/baseline_master_results.csv` match reported numbers exactly.
- [x] Equalized Odds Difference verified as $\max(|\Delta\text{TPR}|, |\Delta\text{FPR}|)$.
- [x] Complete subgroup confusion matrices (TP, TN, FP, FN, TPR, FPR, SR, Approval Rate) computed for `sex` and `age`.
- [x] All fairness metrics independently reconstructed from confusion matrix counts.
- [x] Sensitive attributes (`sex`, `derived_sex`) confirmed excluded from $X$.
- [x] Taiwan `id` confirmed excluded from $X$.
- [x] Preprocessing fitted strictly on training data.
- [x] Zero leakage verified.
- [x] STOPPED before SHAP, fairness mitigation, hyperparameter tuning, or dashboard.

---

## STOPPED HERE

The baseline sanity audit is 100% complete and verified. Awaiting your instructions before proceeding to fairness mitigation or explanation analysis.
