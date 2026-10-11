# Data Leakage Audit Report

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Phase:** Data Audit & Standardization  
**Auditor:** Lead ML Data Engineer  
**Date:** October 2026  

---

## Executive Summary

Data leakage occurs when information from outside the training observation window or directly derived from the target variable is inadvertently supplied to a predictive model. In credit risk scoring and algorithmic fairness research, data leakage not only artificially inflates performance metrics (AUC-ROC, accuracy) but also corrupts model explanations (SHAP values) and invalidates fairness audits.

This audit evaluates all 25 raw columns of the **Taiwan Credit Card Default Dataset** and all 21 raw columns of the **Statlog German Credit Dataset** for temporal, target-derived, and spurious identifier leakage.

---

## Taiwan Credit Card Default Dataset — Leakage Audit

### 1. Column: `id`
- **Reason:** Client record identifier (integers 1 to 30,000). In tree-based models (Random Forest, XGBoost), numerical identifiers can be split upon to partition arbitrary sample subsets, creating spurious memorization.
- **Temporal Alignment:** Administrative sequence number assigned during dataset compilation. Empirical correlation with target is minimal ($r = -0.0139$), but non-causal.
- **Decision:** Exclude from predictive feature matrix $X$; retain in canonical dataset with explicit role `identifier`.
- **Keep/Exclude:** Exclude from $X$.
- **Justification:** Identifiers possess no predictive validity for new applicants. Retaining `id` in the canonical tabular data is mandatory for applicant tracking, record-level auditing, and matched-applicant explanation pair analysis, but it must never be fed into predictive algorithms.

### 2. Columns: `pay_0` (Sept 2005) through `pay_6` (April 2005)
- **Reason:** Repayment status indicators measuring months of payment delay (-2 to 8). Delinquency history is strongly predictive of future default.
- **Temporal Alignment:** All six repayment status variables capture monthly billing cycles preceding the target outcome window (October 2005). Specifically, `pay_0` evaluates the September 2005 payment status, `pay_2` evaluates August 2005, and `pay_6` evaluates April 2005.
- **Decision:** Keep in feature set.
- **Keep/Exclude:** Keep.
- **Justification:** Pre-application or historical behavioral payment performance is standard underwriting criteria. Because the target evaluates default occurring in October 2005, no post-outcome information is present in `pay_0`..`pay_6`.

### 3. Columns: `bill_amt1` (Sept 2005) through `bill_amt6` (April 2005)
- **Reason:** Billed statement amounts in New Taiwan Dollars.
- **Temporal Alignment:** Statements were generated between April 2005 and September 2005.
- **Decision:** Keep in feature set.
- **Keep/Exclude:** Keep.
- **Justification:** Historical statement balances reflect credit utilization prior to the default window. Negative values (-165,580 to -339,603 NT$) represent valid credit balances (account overpayments or merchant refunds) rather than erroneous accounting artifacts.

### 4. Columns: `pay_amt1` (Sept 2005) through `pay_amt6` (April 2005)
- **Reason:** Repayment amounts paid in previous months.
- **Temporal Alignment:** Payments executed between April 2005 and September 2005 to settle previous billing statements.
- **Decision:** Keep in feature set.
- **Keep/Exclude:** Keep.
- **Justification:** Represents actual cash repayments made before October 2005. Does not include post-default collection or debt write-off recoveries.

### 5. Column: `original_target` (`default payment next month`)
- **Reason:** Ground-truth default indicator in October 2005.
- **Decision:** Exclude from feature matrix $X$; retain as ground truth target $y$.
- **Keep/Exclude:** Exclude from $X$.
- **Justification:** Fundamental target variable. Both `original_target` and standardized `target` are preserved in canonical storage for transparency, but excluded from features.

---

## Statlog German Credit Dataset — Leakage Audit

### 1. Column: `checking_status` (Attribute 1)
- **Reason:** Balance of checking account (A11: <0 DM, A12: 0-200 DM, A13: >=200 DM / salary assignments >= 1 yr, A14: no account).
- **Temporal Alignment:** Point-of-application verification of applicant's current liquidity at the lending bank.
- **Decision:** Keep in feature set.
- **Keep/Exclude:** Keep.
- **Justification:** Assesses existing account standing at application time. Does not encode post-loan default management or subsequent account freezes.

### 2. Column: `credit_history` (Attribute 3)
- **Reason:** Past payment history across all existing credits (A30: no credits/paid duly, A31: all at this bank paid duly, A32: existing paid duly till now, A33: past delays, A34: critical account / other credits).
- **Temporal Alignment:** Historical record prior to loan origination.
- **Decision:** Keep in feature set.
- **Keep/Exclude:** Keep.
- **Justification:** Standard credit bureau information. Counter-intuitively, category A34 ("critical account") has high representation among granted loans in the historical dataset because experienced borrowers had higher credit activity; this represents a documented historical underwriting artifact rather than target leakage.

### 3. Column: `original_target` (Attribute 21: Class 1 / 2)
- **Reason:** Cost-matrix class evaluation (1 = Good, 2 = Bad).
- **Decision:** Exclude from feature matrix $X$; mapped to standardized `target` (0 = Good, 1 = Bad).
- **Keep/Exclude:** Exclude from $X$.
- **Justification:** Target outcome variable must never be included in predictive inputs.

---

## Leakage Prevention Protocol for Modeling Pipeline

To guarantee zero leakage during downstream ML experiments, the data engineering pipeline establishes the following contracts:

1. **Feature Separation Contract:** The canonical datasets expose an explicit metadata mapping (`metadata/column_mapping_*.csv`) where every column is tagged with `role`. The modeling pipeline must programmatically select features via:
   $$X = \text{df}[[\text{col for col in df.columns if role\_map[col] in ('feature', 'candidate\_protected\_attribute')} \land \text{col} \neq \text{'id'}]]]$$
2. **Preprocessing Isolation:** No imputation, scaling, encoding, or balance adjustments (SMOTE) have been performed on the canonical datasets. All transformations must be encapsulated inside scikit-learn `Pipeline` objects fitted strictly on the training partition of cross-validation splits.
