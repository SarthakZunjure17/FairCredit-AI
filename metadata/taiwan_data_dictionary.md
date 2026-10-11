# Taiwan Credit Card Default Dataset — Data Dictionary

**Source:** UCI Machine Learning Repository / Chung Hua University & Tamkang University (Yeh & Lien, 2009)

**Total Instances:** 30,000 observations

**Total Variables:** 26 (1 ID, 23 predictive feature candidates, 1 original target, 1 standardized target)

| Column | Meaning | Type | Original Coding | Missing | Role |
| --- | --- | --- | --- | --- | --- |
| id | Unique client record identifier | Numerical (Integer) | 1 to 30000 | 0 (0.0%) | identifier |
| limit_bal | Amount of given credit in NT dollars (individual and supplementary family credit) | Numerical (Continuous) | 10,000 to 1,000,000 NT$ | 0 (0.0%) | feature |
| sex | Gender of credit card client | Categorical (Binary) | 1 = Male (n=11,888, 39.63%); 2 = Female (n=18,112, 60.37%) | 0 (0.0%) | candidate_protected_attribute |
| education | Highest completed education level | Categorical (Multiclass) | 1 = Graduate school (10,585); 2 = University (14,030); 3 = High school (4,917); 4 = Others (123); Undocumented: 0 (14), 5 (280), 6 (51) | 0 (0.0%) [Semantic: 345 in undocumented codes 0/5/6] | feature |
| marriage | Marital status | Categorical (Multiclass) | 1 = Married (13,659); 2 = Single (15,964); 3 = Others (323); Undocumented: 0 (54) | 0 (0.0%) [Semantic: 54 in undocumented code 0] | candidate_protected_attribute |
| age | Age of client in years | Numerical (Discrete / Continuous) | 21 to 79 years (Mean: 35.49, Median: 34.0) | 0 (0.0%) | candidate_protected_attribute |
| pay_0 | Repayment status in September 2005 (most recent repayment history) | Ordinal / Categorical | -2 = Inactive / no balance (2,759); -1 = Paid in full duly (5,686); 0 = Revolving credit active (14,737); 1 = Delay 1 month (3,688); 2 = Delay 2 months (2,667); 3 = Delay 3 months (322); 4 = Delay 4 months (76); 5 = Delay 5 months (26); 6 = Delay 6 months (11); 7 = Delay 7 months (9); 8 = Delay 8 months (19) | 0 (0.0%) | feature |
| pay_2 | Repayment status in August 2005 | Ordinal / Categorical | -2 to 8 (-2: 3782, -1: 6050, 0: 15730, 1: 28, 2: 3927, 3: 326, 4: 99, 5: 25, 6: 12, 7: 20, 8: 1) | 0 (0.0%) | feature |
| pay_3 | Repayment status in July 2005 | Ordinal / Categorical | -2 to 8 (-2: 4085, -1: 5938, 0: 15764, 1: 4, 2: 3819, 3: 240, 4: 76, 5: 21, 6: 23, 7: 27, 8: 3) | 0 (0.0%) | feature |
| pay_4 | Repayment status in June 2005 | Ordinal / Categorical | -2 to 8 (-2: 4348, -1: 5687, 0: 16455, 1: 2, 2: 3159, 3: 180, 4: 69, 5: 35, 6: 5, 7: 58, 8: 2) | 0 (0.0%) | feature |
| pay_5 | Repayment status in May 2005 | Ordinal / Categorical | -2 to 8 (-2: 4546, -1: 5539, 0: 16947, 2: 2626, 3: 178, 4: 84, 5: 17, 6: 4, 7: 58, 8: 1) | 0 (0.0%) | feature |
| pay_6 | Repayment status in April 2005 | Ordinal / Categorical | -2 to 8 (-2: 4895, -1: 5740, 0: 16286, 2: 2766, 3: 184, 4: 49, 5: 13, 6: 19, 7: 46, 8: 2) | 0 (0.0%) | feature |
| bill_amt1 | Amount of bill statement in September 2005 (NT dollar) | Numerical (Continuous) | -165,580 to 964,511 NT$ (Negative values indicate credit overpayment balance) | 0 (0.0%) | feature |
| bill_amt2 | Amount of bill statement in August 2005 (NT dollar) | Numerical (Continuous) | -69,777 to 983,931 NT$ | 0 (0.0%) | feature |
| bill_amt3 | Amount of bill statement in July 2005 (NT dollar) | Numerical (Continuous) | -157,264 to 1,664,089 NT$ | 0 (0.0%) | feature |
| bill_amt4 | Amount of bill statement in June 2005 (NT dollar) | Numerical (Continuous) | -170,000 to 891,586 NT$ | 0 (0.0%) | feature |
| bill_amt5 | Amount of bill statement in May 2005 (NT dollar) | Numerical (Continuous) | -81,334 to 927,171 NT$ | 0 (0.0%) | feature |
| bill_amt6 | Amount of bill statement in April 2005 (NT dollar) | Numerical (Continuous) | -339,603 to 961,664 NT$ | 0 (0.0%) | feature |
| pay_amt1 | Amount of previous payment made in September 2005 (NT dollar) | Numerical (Continuous) | 0 to 873,552 NT$ | 0 (0.0%) | feature |
| pay_amt2 | Amount of previous payment made in August 2005 (NT dollar) | Numerical (Continuous) | 0 to 1,684,259 NT$ | 0 (0.0%) | feature |
| pay_amt3 | Amount of previous payment made in July 2005 (NT dollar) | Numerical (Continuous) | 0 to 896,040 NT$ | 0 (0.0%) | feature |
| pay_amt4 | Amount of previous payment made in June 2005 (NT dollar) | Numerical (Continuous) | 0 to 621,000 NT$ | 0 (0.0%) | feature |
| pay_amt5 | Amount of previous payment made in May 2005 (NT dollar) | Numerical (Continuous) | 0 to 426,529 NT$ | 0 (0.0%) | feature |
| pay_amt6 | Amount of previous payment made in April 2005 (NT dollar) | Numerical (Continuous) | 0 to 528,666 NT$ | 0 (0.0%) | feature |
| original_target | Raw default status indicator in October 2005 | Categorical (Binary) | 0 = No default (23,364, 77.88%); 1 = Default (6,636, 22.12%) | 0 (0.0%) | target |
| target | Standardized research credit default target | Categorical (Binary) | 0 = No default / Good credit (23,364, 77.88%); 1 = Credit default / Bad risk (6,636, 22.12%) | 0 (0.0%) | target |

### Semantic Coding Notes:
- **Repayment Status (pay_0 to pay_6):** Codes `-2` (no consumption / inactive card) and `0` (revolving credit active with minimum payment satisfied) are legitimate banking statuses verified by domain literature.
- **Education Codes (0, 5, 6):** Represent 345 instances not explicitly listed in UCI top-level summary. Preserved intact for transparent audit.
- **Marriage Code (0):** Represents 54 instances of unrecorded marital status. Preserved intact.
- **Negative Bill Amounts:** Reflect valid negative statement balances (account overpayments / refunds / credits).
