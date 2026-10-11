# Data Engineering & Standardization Decisions

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Status:** Canonical Reference Document  
**Auditor:** Lead ML Data Engineer  
**Date:** October 2026  

---

## Overview

This document provides a formal, transparent record of every data engineering, standardization, and cleaning decision applied to the two credit scoring benchmark datasets (**Taiwan Credit Card Default** and **Statlog German Credit**). Every decision is grounded in empirical inspection of the source data, domain literature, and responsible AI experimental requirements.

---

## Decision 1 — Immutable Raw Source Data

- **Context:** The original datasets exist as `default of credit card clients.xls` (Taiwan) and `german.data` / `german.doc` (German).
- **Decision:** Original source files are treated as read-only immutable artifacts. Working copies are ingested into `data/raw/taiwan/` and `data/raw/german/`, leaving the root directories untouched.
- **Reason:** Guarantee absolute reproducibility and provenance. Any pipeline rerun can trace data transformations back to byte-identical source files without risk of data corruption or irreversible modification.

---

## Decision 2 — Target Variable Standardization Convention

- **Context:** The two benchmark datasets employ differing original target encodings:
  - *Taiwan:* `default payment next month` $\in \{0, 1\}$, where 0 = Non-default, 1 = Default.
  - *German:* `class` $\in \{1, 2\}$, where 1 = Good credit, 2 = Bad credit (as documented in `german.doc` Cost Matrix).
- **Decision:** Establish a unified research-wide convention:
  $$\text{target} = 1 \implies \text{Credit Default / Bad Credit / Risk Event}$$
  $$\text{target} = 0 \implies \text{Non-Default / Good Credit / Low Risk}$$
- **Mapping:**
  - **Taiwan:**
    - Original 0 $\rightarrow$ Standardized `target = 0` (23,364 cases, 77.88%)
    - Original 1 $\rightarrow$ Standardized `target = 1` (6,636 cases, 22.12%)
  - **German:**
    - Original 1 (Good) $\rightarrow$ Standardized `target = 0` (700 cases, 70.0%)
    - Original 2 (Bad) $\rightarrow$ Standardized `target = 1` (300 cases, 30.0%)
- **Reason:** Uniform target polarity is essential for computing group fairness metrics (Demographic Parity, Equal Opportunity, Disparate Impact) and calculating SHAP values with consistent positive-class interpretation across both datasets.

---

## Decision 3 — Preservation of Dual Target Columns

- **Context:** Normalizing target names can obscure raw ground truth during auditing.
- **Decision:** Canonical processed datasets retain **both** `original_target` and standardized `target`.
- **Reason:** Eliminates ambiguity during verification, allows automated validation scripts to test transformation fidelity, and ensures backward compatibility for replication studies.

---

## Decision 4 — Customer Identifier Treatment (`id`)

- **Dataset:** Taiwan Credit Card Default (`id`, range 1 to 30,000).
- **Decision:** Tagged with `role = "identifier"`. Retained in canonical dataset (`taiwan_credit_clean.csv`), but explicitly excluded from model input features $X$.
- **Reason:** Customer ID carries no generalizeable predictive information. Retaining it in canonical tables allows tracing individual records, verifying matched-applicant pairs for SHAP explanation consistency, and auditing subgroup predictions.

---

## Decision 5 — Taiwan Undocumented Category Codes (`education`, `marriage`)

- **Context:** 
  - `education` documentation lists codes 1 to 4 (1=Grad school, 2=University, 3=High school, 4=Others). In the raw file, 345 records contain undocumented codes: `0` (14), `5` (280), `6` (51).
  - `marriage` documentation lists codes 1 to 3 (1=Married, 2=Single, 3=Others). In the raw file, 54 records contain undocumented code `0`.
- **Decision:** **Do NOT silently delete, recode, or impute** these values in the canonical dataset. Retain raw integer codes `{0, 1, 2, 3, 4, 5, 6}` and `{0, 1, 2, 3}` as-is in `taiwan_credit_clean.csv`.
- **Reason:** Deleting 399 rows or silently merging them into "others" alters the empirical distribution without explicit research consensus. The canonical dataset must reflect true observations. Downstream grouping or imputation policies are flagged for explicit research determination.

---

## Decision 6 — Taiwan Feature-Duplicate Rows

- **Context:** Checking duplicate rows across all 23 predictive features (excluding `id`) identifies 35 duplicate pairs (70 total rows).
- **Empirical Check:** In 100% of these 35 pairs (35/35), both records share the **exact same target outcome** (primarily inactive cardholders with zero balances and zero payments). Every record possesses a distinct, valid customer `id`.
- **Decision:** Retain all 70 rows in canonical data; do not deduplicate.
- **Reason:** In large retail banking portfolios ($N=30,000$), distinct customers with zero credit utilization naturally possess identical feature representations. Deleting one member of each pair would artificially alter portfolio delinquency statistics and distort class priors.

---

## Decision 7 — Taiwan Repayment Status Codes (`pay_0` through `pay_6`)

- **Context:** The UCI documentation describes `-1 = pay duly` and `1-9 = months delay`, but raw data contains values `-2` (2,759 rows in `pay_0`) and `0` (14,737 rows in `pay_0`).
- **Decision:** Retain codes `-2` and `0` in canonical data without modification.
- **Reason:** Domain literature on this dataset (Yeh & Lien, 2009) confirms banking definitions in Taiwan: `-2` designates an inactive account with no balance/consumption; `0` designates active revolving credit where minimum required payments were satisfied without delinquency. They are valid domain signals.

---

## Decision 8 — Taiwan Negative Bill Amounts (`bill_amt1` to `bill_amt6`)

- **Context:** Negative values appear in statement amounts (e.g., `bill_amt1` minimum is -165,580 NT$).
- **Decision:** Retain all negative values as valid numerical quantities.
- **Reason:** In revolving credit accounts, negative statement balances represent account credit balances resulting from customer overpayments, returned merchandise, or statement refunds. They are financially valid and must not be truncated or set to zero.

---

## Decision 9 — German Domain-Specific "Unknown" Categories

- **Context:** In `german.doc`, several categorical features contain categories denoting "unknown" or "none":
  - `checking_status`: A14 ("no checking account", 394 instances).
  - `savings_status`: A65 ("unknown / no savings account", 183 instances).
  - `property`: A124 ("unknown / no property", 154 instances).
- **Decision:** Retain raw alphanumeric codes (A14, A65, A124) without recoding as generic missing values (`NaN`).
- **Reason:** In credit underwriting, the absence of an account or property is an informative risk signal rather than missing data. Treating them as missing values would trigger unnecessary imputation and distort creditworthiness signals.

---

## Decision 10 — German Personal Status and Sex Conflation (`personal_status_sex`)

- **Context:** Attribute 9 combines gender and marital status into 4 observed categories:
  - `A91`: Male, divorced/separated (50 instances)
  - `A92`: Female, divorced/separated/married (310 instances)
  - `A93`: Male, single (548 instances)
  - `A94`: Male, married/widowed (92 instances)
  - (`A95`: Female, single — 0 instances observed)
- **Decision:** Retain raw `personal_status_sex` with codes A91-A94 in canonical data.
- **Reason:** Retaining raw categories preserves semantic fidelity. We document that `A92` represents all female applicants, while `A91`, `A93`, `A94` represent male applicants. Flagged as `⚠️ RESEARCH DECISION REQUIRED` regarding whether downstream fairness mitigations audit Attribute 9 directly or decompose it into derived binary gender.

---

## Decision 11 — Strict Non-Imputation & Deferred Feature Engineering

- **Context:** Standard machine learning workflows frequently apply one-hot encoding, min-max scaling, standardization, or SMOTE during data ingestion.
- **Decision:** **Strictly prohibited** at this data stage. Canonical datasets remain raw tabular representations with semantic category codes and unscaled numerics.
- **Reason:** Permanent encoding or scaling would prevent clean SHAP explanation generation on original feature spaces, risk data leakage prior to cross-validation splits, and compromise explainability fidelity. All transformations will occur inside scikit-learn `Pipeline` objects.

---

## Decision 12 — Standardized Lowercase Snake_Case Naming

- **Decision:** Rename all columns to clean lowercase snake_case (e.g., `LIMIT_BAL` $\rightarrow$ `limit_bal`, `PAY_0` $\rightarrow$ `pay_0`, `Credit history` $\rightarrow$ `credit_history`), while publishing complete 1:1 mapping tables in `metadata/column_mapping_*.csv`.
- **Reason:** Eliminates whitespace, mixed-case errors, and special characters across heterogeneous OS environments (Windows, Linux) and ML frameworks while preserving full traceability.
