"""
Script to generate Data Dictionaries for Taiwan and German Credit Datasets.
Generates:
- metadata/taiwan_data_dictionary.csv
- metadata/taiwan_data_dictionary.md
- metadata/german_data_dictionary.csv
- metadata/german_data_dictionary.md
"""

import os
import pandas as pd


def generate_dictionaries():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    metadata_dir = os.path.join(project_root, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # 1. Taiwan Data Dictionary
    taiwan_dict = [
        {
            "Column": "id",
            "Meaning": "Unique client record identifier",
            "Type": "Numerical (Integer)",
            "Original Coding": "1 to 30000",
            "Missing": "0 (0.0%)",
            "Role": "identifier"
        },
        {
            "Column": "limit_bal",
            "Meaning": "Amount of given credit in NT dollars (individual and supplementary family credit)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "10,000 to 1,000,000 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "sex",
            "Meaning": "Gender of credit card client",
            "Type": "Categorical (Binary)",
            "Original Coding": "1 = Male (n=11,888, 39.63%); 2 = Female (n=18,112, 60.37%)",
            "Missing": "0 (0.0%)",
            "Role": "candidate_protected_attribute"
        },
        {
            "Column": "education",
            "Meaning": "Highest completed education level",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "1 = Graduate school (10,585); 2 = University (14,030); 3 = High school (4,917); 4 = Others (123); Undocumented: 0 (14), 5 (280), 6 (51)",
            "Missing": "0 (0.0%) [Semantic: 345 in undocumented codes 0/5/6]",
            "Role": "feature"
        },
        {
            "Column": "marriage",
            "Meaning": "Marital status",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "1 = Married (13,659); 2 = Single (15,964); 3 = Others (323); Undocumented: 0 (54)",
            "Missing": "0 (0.0%) [Semantic: 54 in undocumented code 0]",
            "Role": "candidate_protected_attribute"
        },
        {
            "Column": "age",
            "Meaning": "Age of client in years",
            "Type": "Numerical (Discrete / Continuous)",
            "Original Coding": "21 to 79 years (Mean: 35.49, Median: 34.0)",
            "Missing": "0 (0.0%)",
            "Role": "candidate_protected_attribute"
        },
        {
            "Column": "pay_0",
            "Meaning": "Repayment status in September 2005 (most recent repayment history)",
            "Type": "Ordinal / Categorical",
            "Original Coding": "-2 = Inactive / no balance (2,759); -1 = Paid in full duly (5,686); 0 = Revolving credit active (14,737); 1 = Delay 1 month (3,688); 2 = Delay 2 months (2,667); 3 = Delay 3 months (322); 4 = Delay 4 months (76); 5 = Delay 5 months (26); 6 = Delay 6 months (11); 7 = Delay 7 months (9); 8 = Delay 8 months (19)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_2",
            "Meaning": "Repayment status in August 2005",
            "Type": "Ordinal / Categorical",
            "Original Coding": "-2 to 8 (-2: 3782, -1: 6050, 0: 15730, 1: 28, 2: 3927, 3: 326, 4: 99, 5: 25, 6: 12, 7: 20, 8: 1)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_3",
            "Meaning": "Repayment status in July 2005",
            "Type": "Ordinal / Categorical",
            "Original Coding": "-2 to 8 (-2: 4085, -1: 5938, 0: 15764, 1: 4, 2: 3819, 3: 240, 4: 76, 5: 21, 6: 23, 7: 27, 8: 3)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_4",
            "Meaning": "Repayment status in June 2005",
            "Type": "Ordinal / Categorical",
            "Original Coding": "-2 to 8 (-2: 4348, -1: 5687, 0: 16455, 1: 2, 2: 3159, 3: 180, 4: 69, 5: 35, 6: 5, 7: 58, 8: 2)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_5",
            "Meaning": "Repayment status in May 2005",
            "Type": "Ordinal / Categorical",
            "Original Coding": "-2 to 8 (-2: 4546, -1: 5539, 0: 16947, 2: 2626, 3: 178, 4: 84, 5: 17, 6: 4, 7: 58, 8: 1)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_6",
            "Meaning": "Repayment status in April 2005",
            "Type": "Ordinal / Categorical",
            "Original Coding": "-2 to 8 (-2: 4895, -1: 5740, 0: 16286, 2: 2766, 3: 184, 4: 49, 5: 13, 6: 19, 7: 46, 8: 2)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "bill_amt1",
            "Meaning": "Amount of bill statement in September 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "-165,580 to 964,511 NT$ (Negative values indicate credit overpayment balance)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "bill_amt2",
            "Meaning": "Amount of bill statement in August 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "-69,777 to 983,931 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "bill_amt3",
            "Meaning": "Amount of bill statement in July 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "-157,264 to 1,664,089 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "bill_amt4",
            "Meaning": "Amount of bill statement in June 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "-170,000 to 891,586 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "bill_amt5",
            "Meaning": "Amount of bill statement in May 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "-81,334 to 927,171 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "bill_amt6",
            "Meaning": "Amount of bill statement in April 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "-339,603 to 961,664 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_amt1",
            "Meaning": "Amount of previous payment made in September 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "0 to 873,552 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_amt2",
            "Meaning": "Amount of previous payment made in August 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "0 to 1,684,259 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_amt3",
            "Meaning": "Amount of previous payment made in July 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "0 to 896,040 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_amt4",
            "Meaning": "Amount of previous payment made in June 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "0 to 621,000 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_amt5",
            "Meaning": "Amount of previous payment made in May 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "0 to 426,529 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "pay_amt6",
            "Meaning": "Amount of previous payment made in April 2005 (NT dollar)",
            "Type": "Numerical (Continuous)",
            "Original Coding": "0 to 528,666 NT$",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "original_target",
            "Meaning": "Raw default status indicator in October 2005",
            "Type": "Categorical (Binary)",
            "Original Coding": "0 = No default (23,364, 77.88%); 1 = Default (6,636, 22.12%)",
            "Missing": "0 (0.0%)",
            "Role": "target"
        },
        {
            "Column": "target",
            "Meaning": "Standardized research credit default target",
            "Type": "Categorical (Binary)",
            "Original Coding": "0 = No default / Good credit (23,364, 77.88%); 1 = Credit default / Bad risk (6,636, 22.12%)",
            "Missing": "0 (0.0%)",
            "Role": "target"
        }
    ]

    df_taiwan_dict = pd.DataFrame(taiwan_dict)
    df_taiwan_dict.to_csv(os.path.join(metadata_dir, "taiwan_data_dictionary.csv"), index=False)

    def df_to_markdown_table(df):
        headers = list(df.columns)
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = []
        for _, row in df.iterrows():
            row_vals = [str(val).replace("|", "\\|") for val in row]
            row_lines.append("| " + " | ".join(row_vals) + " |")
        return "\n".join([header_line, sep_line] + row_lines)

    with open(os.path.join(metadata_dir, "taiwan_data_dictionary.md"), "w", encoding="utf-8") as f:
        f.write("# Taiwan Credit Card Default Dataset — Data Dictionary\n\n")
        f.write("**Source:** UCI Machine Learning Repository / Chung Hua University & Tamkang University (Yeh & Lien, 2009)\n\n")
        f.write("**Total Instances:** 30,000 observations\n\n")
        f.write("**Total Variables:** 26 (1 ID, 23 predictive feature candidates, 1 original target, 1 standardized target)\n\n")
        f.write(df_to_markdown_table(df_taiwan_dict))
        f.write("\n\n### Semantic Coding Notes:\n")
        f.write("- **Repayment Status (pay_0 to pay_6):** Codes `-2` (no consumption / inactive card) and `0` (revolving credit active with minimum payment satisfied) are legitimate banking statuses verified by domain literature.\n")
        f.write("- **Education Codes (0, 5, 6):** Represent 345 instances not explicitly listed in UCI top-level summary. Preserved intact for transparent audit.\n")
        f.write("- **Marriage Code (0):** Represents 54 instances of unrecorded marital status. Preserved intact.\n")
        f.write("- **Negative Bill Amounts:** Reflect valid negative statement balances (account overpayments / refunds / credits).\n")

    # 2. German Data Dictionary
    german_dict = [
        {
            "Column": "checking_status",
            "Meaning": "Status of existing checking account at bank",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A11: < 0 DM (274); A12: 0 <= ... < 200 DM (269); A13: >= 200 DM / salary assignments for >= 1 yr (63); A14: no checking account (394)",
            "Missing": "0 (0.0%) [Semantic: 394 have no checking account]",
            "Role": "feature"
        },
        {
            "Column": "duration_months",
            "Meaning": "Duration of requested credit in months",
            "Type": "Numerical (Integer)",
            "Original Coding": "4 to 72 months (Mean: 20.90, Median: 18.0)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "credit_history",
            "Meaning": "Past payment history across credits",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A30: no credits taken / all paid back duly (40); A31: all credits at this bank paid duly (49); A32: existing credits paid duly till now (530); A33: delay in paying off in past (88); A34: critical account / other credits existing (293)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "purpose",
            "Meaning": "Purpose of the loan application",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A40: car (new) (234); A41: car (used) (103); A42: furniture/equipment (181); A43: radio/television (280); A44: domestic appliances (12); A45: repairs (22); A46: education (50); A48: retraining (9); A49: business (97); A410: others (12); [A47: vacation - does not exist in dataset]",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "credit_amount",
            "Meaning": "Credit amount requested in Deutsche Mark (DM)",
            "Type": "Numerical (Integer)",
            "Original Coding": "250 to 18,424 DM (Mean: 3271.26, Median: 2319.5)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "savings_status",
            "Meaning": "Total savings account and bonds balance",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A61: < 100 DM (603); A62: 100 <= ... < 500 DM (103); A63: 500 <= ... < 1000 DM (63); A64: >= 1000 DM (48); A65: unknown / no savings account (183)",
            "Missing": "0 (0.0%) [Semantic: 183 unknown / no savings]",
            "Role": "feature"
        },
        {
            "Column": "employment_duration",
            "Meaning": "Duration of present employment tenure",
            "Type": "Categorical (Ordinal)",
            "Original Coding": "A71: unemployed (62); A72: < 1 year (172); A73: 1 <= ... < 4 years (339); A74: 4 <= ... < 7 years (174); A75: >= 7 years (253)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "installment_rate",
            "Meaning": "Installment rate in percentage of disposable income",
            "Type": "Numerical / Ordinal",
            "Original Coding": "1 (136); 2 (231); 3 (157); 4 (476)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "personal_status_sex",
            "Meaning": "Combined demographic encoding of personal/marital status and sex",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A91: male : divorced/separated (50); A92: female : divorced/separated/married (310); A93: male : single (548); A94: male : married/widowed (92); [A95: female : single - 0 records]",
            "Missing": "0 (0.0%)",
            "Role": "candidate_protected_attribute"
        },
        {
            "Column": "other_debtors_guarantors",
            "Meaning": "Co-applicants or guarantors involved in the application",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A101: none (907); A102: co-applicant (41); A103: guarantor (52)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "present_residence_since",
            "Meaning": "Duration at present residential address",
            "Type": "Numerical / Ordinal",
            "Original Coding": "1 (130); 2 (308); 3 (149); 4 (413) years",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "property",
            "Meaning": "Most valuable tangible asset/property owned",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A121: real estate (282); A122: building society savings agreement / life insurance (232); A123: car or other (332); A124: unknown / no property (154)",
            "Missing": "0 (0.0%) [Semantic: 154 unknown / no property]",
            "Role": "feature"
        },
        {
            "Column": "age",
            "Meaning": "Age of applicant in years",
            "Type": "Numerical (Integer)",
            "Original Coding": "19 to 75 years (Mean: 35.55, Median: 33.0)",
            "Missing": "0 (0.0%)",
            "Role": "candidate_protected_attribute"
        },
        {
            "Column": "other_installment_plans",
            "Meaning": "Other installment commitments from banks or retail stores",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A141: bank (139); A142: stores (47); A143: none (814)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "housing",
            "Meaning": "Residential accommodation arrangement",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A151: rent (179); A152: own (713); A153: for free (108)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "existing_credits",
            "Meaning": "Number of existing credits held at this bank",
            "Type": "Numerical (Integer)",
            "Original Coding": "1 (633); 2 (333); 3 (28); 4 (6)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "job",
            "Meaning": "Employment and vocational qualification level",
            "Type": "Categorical (Multiclass)",
            "Original Coding": "A171: unemployed / unskilled - non-resident (22); A172: unskilled - resident (200); A173: skilled employee / official (630); A174: management / self-employed / highly qualified employee / officer (148)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "num_dependents",
            "Meaning": "Number of people being liable to provide maintenance/child support for",
            "Type": "Numerical (Integer)",
            "Original Coding": "1 (845); 2 (155)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "telephone",
            "Meaning": "Whether landline telephone is registered in customer name",
            "Type": "Categorical (Binary)",
            "Original Coding": "A191: none (596); A192: yes, registered under customer name (404)",
            "Missing": "0 (0.0%)",
            "Role": "feature"
        },
        {
            "Column": "foreign_worker",
            "Meaning": "Whether the applicant is classified as a foreign worker",
            "Type": "Categorical (Binary)",
            "Original Coding": "A201: yes (963, 96.3%); A202: no (37, 3.7%)",
            "Missing": "0 (0.0%)",
            "Role": "candidate_protected_attribute"
        },
        {
            "Column": "original_target",
            "Meaning": "Original Statlog credit outcome from german.doc / german.data",
            "Type": "Categorical (Binary)",
            "Original Coding": "1 = Good Credit (700, 70.0%); 2 = Bad Credit (300, 30.0%)",
            "Missing": "0 (0.0%)",
            "Role": "target"
        },
        {
            "Column": "target",
            "Meaning": "Standardized research credit default target",
            "Type": "Categorical (Binary)",
            "Original Coding": "0 = Good Credit / Non-default (700, 70.0%); 1 = Bad Credit / Risk Event (300, 30.0%)",
            "Missing": "0 (0.0%)",
            "Role": "target"
        }
    ]

    df_german_dict = pd.DataFrame(german_dict)
    df_german_dict.to_csv(os.path.join(metadata_dir, "german_data_dictionary.csv"), index=False)

    with open(os.path.join(metadata_dir, "german_data_dictionary.md"), "w", encoding="utf-8") as f:
        f.write("# Statlog German Credit Dataset — Data Dictionary\n\n")
        f.write("**Source:** Professor Dr. Hans Hofmann, Universitaet Hamburg (1994), via UCI ML Repository\n\n")
        f.write("**Total Instances:** 1,000 observations\n\n")
        f.write("**Total Variables:** 22 (20 predictive feature candidates, 1 original target, 1 standardized target)\n\n")
        f.write(df_to_markdown_table(df_german_dict))
        f.write("\n\n### Semantic Coding Notes:\n")
        f.write("- **Personal Status and Sex (Attribute 9):** Conflates sex and marital status into 4 observed categories (A91: male divorced/sep, A92: female div/sep/mar, A93: male single, A94: male married/widowed). Category A95 (female single) is absent in this dataset.\n")
        f.write("- **Target Standardization Mapping:** Original code 1 (Good) maps to standardized `0` (non-risk); original code 2 (Bad) maps to standardized `1` (risk event / default).\n")
        f.write("- **Purpose A47:** `german.doc` notes `A47 : (vacation - does not exist?)`. The dataset contains exactly 0 instances of A47, confirming the documentation notation.\n")

    print("Successfully generated:")
    print(f"  - {os.path.join(metadata_dir, 'taiwan_data_dictionary.csv')}")
    print(f"  - {os.path.join(metadata_dir, 'taiwan_data_dictionary.md')}")
    print(f"  - {os.path.join(metadata_dir, 'german_data_dictionary.csv')}")
    print(f"  - {os.path.join(metadata_dir, 'german_data_dictionary.md')}")


if __name__ == "__main__":
    generate_dictionaries()
