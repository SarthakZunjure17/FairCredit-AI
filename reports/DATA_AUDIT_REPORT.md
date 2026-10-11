# Comprehensive Data Audit Report

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Lead ML Data Engineer:** Responsible AI Engineering Team  
**Evaluation Date:** October 2026  
**Audit Protocol:** Pre-Modeling Data Verification, Quality Profiling & Standardization  

---

## 1. Executive Summary & Audit Status

Before undertaking machine learning benchmarking (Logistic Regression, Random Forest, XGBoost), fairness mitigation (Fairlearn), or explanation consistency auditing (SHAP), a rigorous pre-modeling data audit was executed across the two core benchmark credit datasets:
1. **Taiwan Credit Card Default Dataset** (Primary Benchmark, $N=30,000$)
2. **Statlog German Credit Dataset** (Secondary Validation Benchmark, $N=1,000$)

### Overall Audit Status
$$\mathbf{DATA\ AUDIT\ STATUS:\ PASS\ WITH\ RESEARCH\ WARNINGS}$$

All critical data pipeline contracts—raw immutability, target polarity unification ($y=1$ as Default/Bad Credit), identifier exclusion, schema consistency, zero inadvertent data leakage, and automated verification suite—have been **fully verified and passed (100%)**.

Warnings are raised regarding domain-specific demographic codings (e.g., German conflated sex/marital status, near-constant foreign worker distribution, and Taiwan undocumented categorical codes {0, 5, 6} in education), which are formally flagged for research decision-making rather than altered silently.

---

## 2. Raw Source Files Inspected

In adherence to **Rule 1 (Never Modify Raw Data)**, the original files were inspected in place and preserved without modification:

| Dataset | Raw Source Location | File Format | File Size | Ingested Raw Destination |
| :--- | :--- | :--- | :--- | :--- |
| **Taiwan** | `default+of+credit+card+clients (1)/default of credit card clients.xls` | Excel 97-2003 (BIFF8 OLE2) | 5,539,328 bytes | `data/raw/taiwan/default of credit card clients.xls` |
| **German** | `statlog+german+credit+data/german.data` | Space-delimited ASCII | 79,793 bytes | `data/raw/german/german.data` |
| **German** | `statlog+german+credit+data/german.doc` | Plain Text Documentation | 4,679 bytes | `data/raw/german/german.doc` |
| **German** | `statlog+german+credit+data/Index` | Plain Text Metadata | 150 bytes | `data/raw/german/Index` |
| **German** | `statlog+german+credit+data/german.data-numeric` | Strathclyde preprocessed numeric | 102,000 bytes | `data/raw/german/german.data-numeric` (archival only) |

> [!NOTE]
> Per Rule 3, `german.data-numeric` was explicitly **not** used as the modeling foundation because it strips categorical semantics. All German features were parsed directly from `german.data` using `german.doc`.

---

## 3. Dataset Dimensions & Inspection Findings

### 3.1 Taiwan Credit Card Default Dataset
- **Excel Structure:** Sheet name `'Data'`, dimensions 30,002 rows $\times$ 25 columns.
  - Row 0 contains UCI variable aliases: `[NaN, 'X1', 'X2', ..., 'X23', 'Y']`.
  - Row 1 contains actual feature headers: `['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', ..., 'default payment next month']`.
  - Rows 2 through 30,001 contain observed customer data.
- **Actual Observations ($N$):** 30,000 credit records.
- **Raw Variables ($P$):** 25 columns (1 ID, 23 feature candidates, 1 target).
- **Canonical Processed Dimensions:** 30,000 rows $\times$ 26 columns (includes both `original_target` and `target`).

### 3.2 Statlog German Credit Dataset
- **File Structure:** Space-delimited table with no header row.
- **Actual Observations ($N$):** 1,000 credit records.
- **Raw Variables ($P$):** 21 columns (20 attributes + 1 class variable).
- **Canonical Processed Dimensions:** 1,000 rows $\times$ 22 columns (20 features + `original_target` + `target`).

---

## 4. Target Variable Standardization

In algorithmic fairness and explainability research, heterogeneous target polarities lead to reversed confusion matrices, contradictory fairness metrics (e.g., disparate impact calculated on approvals vs rejections), and inverted SHAP contribution directions. We establish a strict internal standard:

$$\mathbf{y = 1 \implies CREDIT\ DEFAULT\ /\ BAD\ CREDIT\ /\ RISK\ EVENT}$$
$$\mathbf{y = 0 \implies NO\ DEFAULT\ /\ GOOD\ CREDIT\ /\ APPROVAL\ OUTCOME}$$

### Target Mapping Summary:
| Dataset | Original Column Name | Raw Target Encoding | Standardized Mapping | Class Distribution ($y=0$ : $y=1$) | Default Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Taiwan** | `default payment next month` | `0` = No default<br>`1` = Default | $0 \rightarrow 0$<br>$1 \rightarrow 1$ | $23,364 : 6,636$ | **22.12%** |
| **German** | Attribute 21 (`class`) | `1` = Good Credit<br>`2` = Bad Credit | $1 \rightarrow 0$<br>$2 \rightarrow 1$ | $700 : 300$ | **30.00%** |

Both `original_target` and standardized `target` are retained in the canonical datasets to preserve total data provenance.

---

## 5. Customer Identifier Analysis

- **Taiwan Credit:** Contains column `ID` (integers 1 to 30,000).
  - Uniqueness: 30,000 unique values (0 duplicate IDs).
  - Target correlation: $r = -0.013952$.
  - Classification: `role = "identifier"`. Excluded from model features $X$.
  - Purpose in Pipeline: Tracing records, tracking matched-applicant explanation consistency pairs in SHAP audits.
- **German Credit:** Contains **no identifier** in the raw source data (only 20 attributes and target).
  - Classification: None in raw file. Implicit row index (0 to 999) serves as record pointer.

---

## 6. Candidate Protected / Sensitive Attributes

In this Responsible AI study, demographic attributes serve as the audit basis for fairness metrics (Demographic Parity Difference, Equal Opportunity Difference, Disparate Impact). They must be isolated from standard features and treated with explicit experimental policies.

### 6.1 Taiwan Dataset Candidate Protected Attributes

| Feature | Role | Original Encoding | Group Counts | Group % | Empirical Default Rate ($y=1$) | Disparity / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`sex`** | `candidate_protected_attribute` | 1 = Male<br>2 = Female | Male: 11,888<br>Female: 18,112 | 39.63%<br>60.37% | Male: **24.17%**<br>Female: **20.78%** | Male default rate is +3.39% higher |
| **`age`** | `candidate_protected_attribute` | Continuous (21 to 79)<br>Mean: 35.49, Median: 34 | $<25$: 3,871<br>$25-35$: 12,938<br>$35-50$: 10,922<br$>50$: 2,269 | 12.90%<br>43.13%<br>36.41%<br>7.56% | $<25$: **26.66%**<br>$25-35$: **19.83%**<br>$35-50$: **22.54%**<br$>50$: **25.43%** | Young borrowers ($<25$) exhibit highest default rate |
| **`marriage`** | `candidate_protected_attribute` | 1 = Married<br>2 = Single<br>3 = Others<br>0 = Undocumented | Married: 13,659<br>Single: 15,964<br>Others: 323<br>Undoc: 54 | 45.53%<br>53.21%<br>1.08%<br>0.18% | Married: **23.47%**<br>Single: **20.93%**<br>Others: **26.01%**<br>Undoc: **9.26%** | Single borrowers have lower default rates |

### 6.2 German Dataset Candidate Protected Attributes

| Feature | Role | Original Encoding | Group Counts | Group % | Empirical Bad Credit Rate ($y=1$) | Disparity / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`personal_status_sex`** | `candidate_protected_attribute` | Attribute 9:<br>A91 = Male: div/sep<br>A92 = Female: div/sep/mar<br>A93 = Male: single<br>A94 = Male: mar/wid<br>(A95 = Female: single, n=0) | A91: 50<br>A92: 310<br>A93: 548<br>A94: 92 | 5.0%<br>31.0%<br>54.8%<br>9.2% | A91: **40.00%**<br>A92: **35.16%**<br>A93: **26.64%**<br>A94: **27.17%** | **Derived Gender:**<br>Female (A92): **35.16%**<br>Male (A91+A93+A94): **27.68%**<br>($\Delta = +7.48\%$ penalty for women) |
| **`age`** | `candidate_protected_attribute` | Attribute 13: Continuous (19 to 75)<br>Mean: 35.55, Median: 33 | $\le 25$: 190<br>$>25$: 810 | 19.0%<br>81.0% | $\le 25$: **42.11%**<br>$>25$: **27.16%** | Severe age penalty: applicants $\le 25$ have **+14.95%** higher bad credit rate |
| **`foreign_worker`** | `candidate_protected_attribute` | Attribute 20:<br>A201 = Foreign worker<br>A202 = Non-foreign worker | A201: 963<br>A202: 37 | 96.3%<br>3.7% | A201: **30.74%**<br>A202: **10.81%** | Significant outcome disparity, but severe class imbalance (only 37 non-foreign) |

---

## 7. Data Quality Audit Findings

### 7.1 Missing Values Audit
- **Explicit Missing Values (`NaN`, null, empty cells):**
  - Taiwan Dataset: **0** across all 30,000 rows and 26 columns.
  - German Dataset: **0** across all 1,000 rows and 22 columns.
- **Implicit Domain "Unknown" Values:**
  - German `checking_status`: Category `A14` represents "no checking account" (394 records, 39.4%).
  - German `savings_status`: Category `A65` represents "unknown / no savings account" (183 records, 18.3%).
  - German `property`: Category `A124` represents "unknown / no property" (154 records, 15.4%).
  - Taiwan `education`: Undocumented categories `0` (14), `5` (280), `6` (51) represent 345 records (1.15%).
  - Taiwan `marriage`: Undocumented category `0` represents 54 records (0.18%).
- **Policy Decision:** No values were imputed. Domain codes were preserved in raw semantic form.

### 7.2 Duplicate Records Audit
- **Exact Full-Row Duplicates:**
  - Taiwan: **0** exact full-row duplicates.
  - German: **0** exact full-row duplicates.
- **Feature Duplicates (ignoring ID):**
  - Taiwan: **35 duplicate pairs (70 records total)**.
    - Inspection: In 100% of these 35 pairs (35/35), the records share **identical target values** ($y=0$ in 33 pairs, $y=1$ in 2 pairs).
    - Profile: Primarily inactive cards with 0 credit limit usage, 0 billing statements, and 0 payments.
    - Decision: Retain all 70 rows because each represents an authentic, independent customer account ($ID_1 \neq ID_2$). Deleting them would artificially alter portfolio delinquency statistics.
  - German: **0** feature duplicates.

### 7.3 Invalid or Suspicious Category Codes
- **Taiwan Repayment Statuses (`pay_0` to `pay_6`):**
  - Observed range: `[-2, 8]`.
  - Categories `-2` (inactive/zero balance) and `0` (active revolving credit with minimum payment satisfied) were verified as valid domain codes in Taiwan banking practice (Yeh & Lien, 2009).
- **German Purpose (`purpose`):**
  - Code `A47` ("vacation") is listed in `german.doc` with the comment `(vacation - does not exist?)`.
  - Inspection confirms **0 instances** of `A47` exist in `german.data`.
- **German Personal Status (`personal_status_sex`):**
  - Code `A95` ("female : single") has **0 instances** in `german.data`. All females are aggregated into `A92`.

### 7.4 Numerical Sanity & Outlier Checks
- **Taiwan Bill Amounts (`bill_amt1` to `bill_amt6`):**
  - Minimum values are negative (e.g., `bill_amt1` min = -165,580 NT$).
  - Verification: Negative statement balances represent credit card customer overpayments or statement credits/refunds. Financially valid; preserved as continuous numbers.
- **Taiwan Payment Amounts (`pay_amt1` to `pay_amt6`):**
  - All values are non-negative ($0.0$ to $1,684,259.0$ NT$). Realistic.
- **German Credit Amounts (`credit_amount`):**
  - Range: 250 to 18,424 DM. Highly right-skewed (median 2,319.5 DM, mean 3,271.26 DM). Preserved unscaled.
- **German Loan Duration (`duration_months`):**
  - Range: 4 to 72 months (median 18.0 months). Realistic.

---

## 8. Data Leakage Audit Summary

A dedicated audit ([`reports/DATA_LEAKAGE_AUDIT.md`](file:///d:/sarthak/Subjects/ERA/Project/reports/DATA_LEAKAGE_AUDIT.md)) confirmed:
1. **Temporal Coherence:** In Taiwan credit data, all feature observations (`pay_0`..`pay_6`, `bill_amt1`..`bill_amt6`, `pay_amt1`..`pay_amt6`) occurred strictly between April 2005 and September 2005, antecedent to the target evaluation month (October 2005).
2. **Identifier Separation:** The `id` column is tagged with `role = "identifier"` and programmatically excluded from predictive feature matrices $X$.
3. **Target Cleanliness:** Ground-truth targets (`original_target`, `target`) are isolated from feature inputs.

---

## 9. Final Canonical Dataset Specifications

The pipeline generated canonical, clean datasets with verified schemas:

### 9.1 Taiwan Credit Card Clean Dataset
- **CSV:** [`data/processed/taiwan_credit_clean.csv`](file:///d:/sarthak/Subjects/ERA/Project/data/processed/taiwan_credit_clean.csv) (3,737,205 bytes)
- **Parquet:** [`data/processed/taiwan_credit_clean.parquet`](file:///d:/sarthak/Subjects/ERA/Project/data/processed/taiwan_credit_clean.parquet) (1,824,553 bytes)
- **Dimensions:** 30,000 rows $\times$ 26 columns
- **Column Schema:**
  - Identifier (1): `id` (int64)
  - Features (23): `limit_bal` (float64), `sex` (int64), `education` (int64), `marriage` (int64), `age` (int64), `pay_0` (int64), `pay_2` (int64), `pay_3` (int64), `pay_4` (int64), `pay_5` (int64), `pay_6` (int64), `bill_amt1`..`bill_amt6` (float64), `pay_amt1`..`pay_amt6` (float64)
  - Target (2): `original_target` (int64), `target` (int64)

### 9.2 German Credit Clean Dataset
- **CSV:** [`data/processed/german_credit_clean.csv`](file:///d:/sarthak/Subjects/ERA/Project/data/processed/german_credit_clean.csv) (83,113 bytes)
- **Parquet:** [`data/processed/german_credit_clean.parquet`](file:///d:/sarthak/Subjects/ERA/Project/data/processed/german_credit_clean.parquet) (25,454 bytes)
- **Dimensions:** 1,000 rows $\times$ 22 columns
- **Column Schema:**
  - Categorical Features (13): `checking_status` (object), `credit_history` (object), `purpose` (object), `savings_status` (object), `employment_duration` (object), `personal_status_sex` (object), `other_debtors_guarantors` (object), `property` (object), `other_installment_plans` (object), `housing` (object), `job` (object), `telephone` (object), `foreign_worker` (object)
  - Numerical Features (7): `duration_months` (int64), `credit_amount` (int64), `installment_rate` (int64), `present_residence_since` (int64), `age` (int64), `existing_credits` (int64), `num_dependents` (int64)
  - Target (2): `original_target` (int64), `target` (int64)

---

## 10. Automated Validation Results

Running `python validation/validate_all.py` executes 22 automated assertions:

```text
============================================================
MASTER DATA VALIDATION SUITE RESULTS
============================================================
Taiwan Dataset Validation: PASS WITH WARNINGS
  - Immutability check: PASS
  - Dimensions (30000 x 26): PASS
  - Target binary {0, 1} & counts (23364 : 6636): PASS
  - Unique ID & role assignment: PASS
  - Lowercase snake_case naming: PASS
  - Zero NaNs: PASS
  - Value ranges & sanity: PASS

German Dataset Validation: PASS WITH WARNINGS
  - Immutability check: PASS
  - Documentation parsed: PASS
  - Dimensions (1000 x 22): PASS
  - Target binary {0, 1} & counts (700 : 300): PASS
  - Mapping (1->0, 2->1): PASS
  - Lowercase snake_case naming: PASS
  - Zero NaNs: PASS
  - Categorical codes against german.doc: PASS

OVERALL RESULT: PASS WITH RESEARCH WARNINGS (Exit Code 0)
```

---

## 11. Research Decisions Requiring Project Team Confirmation

In accordance with **Phase 19**, the following five design ambiguities must **NOT** be decided silently. They are formally submitted for project team decision:

### ⚠️ RESEARCH DECISION 1 — German Gender Extraction from Conflated Attribute 9
- **Problem:** In German Credit, Attribute 9 (`personal_status_sex`) merges gender and marital status into `A91` (male div/sep), `A92` (female div/sep/mar), `A93` (male single), `A94` (male mar/wid).
- **Options:**
  - **Option A (Standard Benchmark Convention):** Decompose Attribute 9 into an explicit binary demographic feature: `sex = 'female'` if $A92$ else `'male'`, retaining marital status separately. This aligns with Kozodoi et al. (2022) and AIF360.
  - **Option B (Preserve Conflated Encoding):** Keep Attribute 9 as a 4-level categorical variable for both feature and fairness auditing.
- **Recommendation:** **Option A**. Decomposing into binary sex allows direct calculation of group fairness metrics on female vs. male applicants.

---

### ⚠️ RESEARCH DECISION 2 — Inclusion vs. Exclusion of Protected Attributes in Predictive Feature Matrix $X$
- **Problem:** When benchmarking models for fairness and explanation consistency, should protected attributes (`sex`, `age`) be included in the predictive feature matrix $X$ fed to Logistic Regression, Random Forest, and XGBoost ("Fairness through Awareness"), or blinded from $X$ ("Fairness through Unawareness")?
- **Options:**
  - **Option A (Fairness through Awareness):** Include demographic attributes in $X$ during baseline training, allowing models to learn overt weights that in-processing mitigations can penalize and SHAP can explain directly.
  - **Option B (Fairness through Unawareness):** Exclude demographic attributes from $X$ for all predictive models, retaining them solely in test partitions for fairness auditing and matched-applicant explanation pair evaluation.
- **Recommendation:** **Option B (or dual-branch benchmarking)**. Under ECOA regulation, credit models cannot use demographic attributes; comparing explanation consistency across both configurations would make a novel contribution to the research paper.

---

### ⚠️ RESEARCH DECISION 3 — Age Discretization Threshold for Algorithmic Fairness Audits
- **Problem:** `age` is a continuous numerical variable in both datasets (Taiwan: 21–79; German: 19–75). Group fairness metrics (e.g., Demographic Parity, Equal Opportunity) require binary or discrete group definitions.
- **Options:**
  - **Option A (Standard Benchmark Threshold):** Define privileged group as $\text{age} > 25$ and unprivileged as $\text{age} \le 25$ (consistent with German Credit literature: Kamiran & Calders, 2012; Kozodoi et al., 2022).
  - **Option B (Domain Banking Age Threshold):** Define unprivileged group as $\text{age} < 30$ or senior applicants $\text{age} \ge 60$.
- **Recommendation:** **Option A** for German dataset ($\le 25$ vs $>25$), and evaluate both $\le 25$ and age quartiles for Taiwan.

---

### ⚠️ RESEARCH DECISION 4 — Taiwan Undocumented Category Codes Treatment (`education`, `marriage`)
- **Problem:** Taiwan `education` has 345 instances of undocumented codes $\{0, 5, 6\}$, and `marriage` has 54 instances of code $0$.
- **Options:**
  - **Option A (Grouping into 'Others'):** In the modeling pipeline's preprocessing stage, map $\{0, 5, 6\} \rightarrow 4$ (Others) in `education`, and $0 \rightarrow 3$ (Others) in `marriage`.
  - **Option B (Separate 'Unknown' Category):** Treat $\{0, 5, 6\}$ as a distinct categorical bucket `'unknown'` in one-hot encoding.
  - **Option C (Drop Records):** Drop the 399 rows (1.33% of dataset).
- **Recommendation:** **Option A**. Merging into existing "Others" preserves 100% of observations and aligns with common practice in published credit scoring literature.

---

### ⚠️ RESEARCH DECISION 5 — Suitability of German `foreign_worker` for Fairness Auditing
- **Problem:** Attribute 20 (`foreign_worker`) has a 96.3% vs 3.7% distribution (963 foreign vs 37 non-foreign).
- **Options:**
  - **Option A:** Include `foreign_worker` in secondary intersectional fairness evaluations, but note extreme imbalance.
  - **Option B:** Exclude `foreign_worker` from primary fairness experiments to prevent high variance in group-level metric calculations.
- **Recommendation:** **Option B**. Focus primary fairness mitigation and explanation consistency analysis on `sex` and `age`.
