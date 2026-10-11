# Statlog German Credit Dataset — Data Dictionary

**Source:** Professor Dr. Hans Hofmann, Universitaet Hamburg (1994), via UCI ML Repository

**Total Instances:** 1,000 observations

**Total Variables:** 22 (20 predictive feature candidates, 1 original target, 1 standardized target)

| Column | Meaning | Type | Original Coding | Missing | Role |
| --- | --- | --- | --- | --- | --- |
| checking_status | Status of existing checking account at bank | Categorical (Multiclass) | A11: < 0 DM (274); A12: 0 <= ... < 200 DM (269); A13: >= 200 DM / salary assignments for >= 1 yr (63); A14: no checking account (394) | 0 (0.0%) [Semantic: 394 have no checking account] | feature |
| duration_months | Duration of requested credit in months | Numerical (Integer) | 4 to 72 months (Mean: 20.90, Median: 18.0) | 0 (0.0%) | feature |
| credit_history | Past payment history across credits | Categorical (Multiclass) | A30: no credits taken / all paid back duly (40); A31: all credits at this bank paid duly (49); A32: existing credits paid duly till now (530); A33: delay in paying off in past (88); A34: critical account / other credits existing (293) | 0 (0.0%) | feature |
| purpose | Purpose of the loan application | Categorical (Multiclass) | A40: car (new) (234); A41: car (used) (103); A42: furniture/equipment (181); A43: radio/television (280); A44: domestic appliances (12); A45: repairs (22); A46: education (50); A48: retraining (9); A49: business (97); A410: others (12); [A47: vacation - does not exist in dataset] | 0 (0.0%) | feature |
| credit_amount | Credit amount requested in Deutsche Mark (DM) | Numerical (Integer) | 250 to 18,424 DM (Mean: 3271.26, Median: 2319.5) | 0 (0.0%) | feature |
| savings_status | Total savings account and bonds balance | Categorical (Multiclass) | A61: < 100 DM (603); A62: 100 <= ... < 500 DM (103); A63: 500 <= ... < 1000 DM (63); A64: >= 1000 DM (48); A65: unknown / no savings account (183) | 0 (0.0%) [Semantic: 183 unknown / no savings] | feature |
| employment_duration | Duration of present employment tenure | Categorical (Ordinal) | A71: unemployed (62); A72: < 1 year (172); A73: 1 <= ... < 4 years (339); A74: 4 <= ... < 7 years (174); A75: >= 7 years (253) | 0 (0.0%) | feature |
| installment_rate | Installment rate in percentage of disposable income | Numerical / Ordinal | 1 (136); 2 (231); 3 (157); 4 (476) | 0 (0.0%) | feature |
| personal_status_sex | Combined demographic encoding of personal/marital status and sex | Categorical (Multiclass) | A91: male : divorced/separated (50); A92: female : divorced/separated/married (310); A93: male : single (548); A94: male : married/widowed (92); [A95: female : single - 0 records] | 0 (0.0%) | candidate_protected_attribute |
| other_debtors_guarantors | Co-applicants or guarantors involved in the application | Categorical (Multiclass) | A101: none (907); A102: co-applicant (41); A103: guarantor (52) | 0 (0.0%) | feature |
| present_residence_since | Duration at present residential address | Numerical / Ordinal | 1 (130); 2 (308); 3 (149); 4 (413) years | 0 (0.0%) | feature |
| property | Most valuable tangible asset/property owned | Categorical (Multiclass) | A121: real estate (282); A122: building society savings agreement / life insurance (232); A123: car or other (332); A124: unknown / no property (154) | 0 (0.0%) [Semantic: 154 unknown / no property] | feature |
| age | Age of applicant in years | Numerical (Integer) | 19 to 75 years (Mean: 35.55, Median: 33.0) | 0 (0.0%) | candidate_protected_attribute |
| other_installment_plans | Other installment commitments from banks or retail stores | Categorical (Multiclass) | A141: bank (139); A142: stores (47); A143: none (814) | 0 (0.0%) | feature |
| housing | Residential accommodation arrangement | Categorical (Multiclass) | A151: rent (179); A152: own (713); A153: for free (108) | 0 (0.0%) | feature |
| existing_credits | Number of existing credits held at this bank | Numerical (Integer) | 1 (633); 2 (333); 3 (28); 4 (6) | 0 (0.0%) | feature |
| job | Employment and vocational qualification level | Categorical (Multiclass) | A171: unemployed / unskilled - non-resident (22); A172: unskilled - resident (200); A173: skilled employee / official (630); A174: management / self-employed / highly qualified employee / officer (148) | 0 (0.0%) | feature |
| num_dependents | Number of people being liable to provide maintenance/child support for | Numerical (Integer) | 1 (845); 2 (155) | 0 (0.0%) | feature |
| telephone | Whether landline telephone is registered in customer name | Categorical (Binary) | A191: none (596); A192: yes, registered under customer name (404) | 0 (0.0%) | feature |
| foreign_worker | Whether the applicant is classified as a foreign worker | Categorical (Binary) | A201: yes (963, 96.3%); A202: no (37, 3.7%) | 0 (0.0%) | candidate_protected_attribute |
| original_target | Original Statlog credit outcome from german.doc / german.data | Categorical (Binary) | 1 = Good Credit (700, 70.0%); 2 = Bad Credit (300, 30.0%) | 0 (0.0%) | target |
| target | Standardized research credit default target | Categorical (Binary) | 0 = Good Credit / Non-default (700, 70.0%); 1 = Bad Credit / Risk Event (300, 30.0%) | 0 (0.0%) | target |

### Semantic Coding Notes:
- **Personal Status and Sex (Attribute 9):** Conflates sex and marital status into 4 observed categories (A91: male divorced/sep, A92: female div/sep/mar, A93: male single, A94: male married/widowed). Category A95 (female single) is absent in this dataset.
- **Target Standardization Mapping:** Original code 1 (Good) maps to standardized `0` (non-risk); original code 2 (Bad) maps to standardized `1` (risk event / default).
- **Purpose A47:** `german.doc` notes `A47 : (vacation - does not exist?)`. The dataset contains exactly 0 instances of A47, confirming the documentation notation.
