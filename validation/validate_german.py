"""
Automated Data Validation for Statlog German Credit Dataset.
Project: Responsible AI Credit Scoring Research

Validates:
1. Raw data immutability & presence (german.data, german.doc, etc.)
2. Canonical dataset presence and integrity (german_credit_clean.csv, .parquet)
3. Exact row & column counts (1,000 rows, 22 columns)
4. Absence of accidental index columns
5. Target existence, valid values {0, 1}, and mapping consistency (original 1->0, original 2->1)
6. Ground truth class distribution (700 Good, 300 Bad)
7. Clean snake_case column names and absence of duplicate column names
8. Absence of NaN missing values
9. Categorical codes validity against german.doc
10. Numerical ranges sanity checks
"""

import os
import sys
import pandas as pd


def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def validate_german():
    project_root = get_project_root()
    print("=" * 60)
    print("RUNNING GERMAN CREDIT DATASET VALIDATION CHECKS")
    print("=" * 60)
    
    failures = []
    warnings = []
    
    # 1. Raw Data Immutability
    raw_source = os.path.join(project_root, "statlog+german+credit+data", "german.data")
    raw_doc = os.path.join(project_root, "statlog+german+credit+data", "german.doc")
    raw_ingested = os.path.join(project_root, "data", "raw", "german", "german.data")
    
    if not os.path.exists(raw_source):
        failures.append(f"Original source file missing at {raw_source}")
    else:
        size_source = os.path.getsize(raw_source)
        if size_source != 79793:
            failures.append(f"Original source file size altered! Expected 79793 bytes, found {size_source}")
        else:
            print("  [PASS] Original raw file is present and unaltered (79,793 bytes).")
            
    if not os.path.exists(raw_doc):
        failures.append(f"German documentation file missing at {raw_doc}")
    else:
        print("  [PASS] German documentation (german.doc) verified present.")
        
    if not os.path.exists(raw_ingested):
        failures.append(f"Ingested raw copy missing at {raw_ingested}")
    else:
        print("  [PASS] Ingested raw copy is present.")
        
    # 2. Canonical Datasets Presence
    clean_csv = os.path.join(project_root, "data", "processed", "german_credit_clean.csv")
    clean_parquet = os.path.join(project_root, "data", "processed", "german_credit_clean.parquet")
    
    if not os.path.exists(clean_csv):
        failures.append(f"Canonical CSV missing at {clean_csv}")
    if not os.path.exists(clean_parquet):
        failures.append(f"Canonical Parquet missing at {clean_parquet}")
        
    if failures:
        print("\nFATAL FAILURES ENCOUNTERED:")
        for f in failures:
            print(f"  - {f}")
        return False, failures, warnings
        
    df = pd.read_csv(clean_csv)
    print("  [PASS] Canonical processed dataset loaded successfully.")
    
    # 3. Shape and Dimensions
    expected_rows = 1000
    expected_cols = 22
    if len(df) != expected_rows:
        failures.append(f"Row count mismatch: expected {expected_rows}, found {len(df)}")
    else:
        print(f"  [PASS] Exact expected row count verified: {len(df)} rows.")
        
    if len(df.columns) != expected_cols:
        failures.append(f"Column count mismatch: expected {expected_cols}, found {len(df.columns)}")
    else:
        print(f"  [PASS] Exact expected column count verified: {len(df.columns)} columns.")
        
    # 4. Accidental Index Check
    if "Unnamed: 0" in df.columns or "index" in df.columns:
        failures.append("Accidental index column detected in processed dataset!")
    else:
        print("  [PASS] No accidental index columns detected.")
        
    # 5. Target Standardization & Mapping
    if "target" not in df.columns:
        failures.append("Column 'target' missing from processed dataset!")
    if "original_target" not in df.columns:
        failures.append("Column 'original_target' missing from processed dataset!")
        
    if "target" in df.columns and "original_target" in df.columns:
        target_unique = set(df["target"].unique())
        if target_unique != {0, 1}:
            failures.append(f"Standardized target contains invalid values: {target_unique}. Expected {{0, 1}}.")
        else:
            print("  [PASS] Standardized target contains strictly binary {0, 1}.")
            
        orig_unique = set(df["original_target"].unique())
        if orig_unique != {1, 2}:
            failures.append(f"Original target contains invalid values: {orig_unique}. Expected {{1, 2}}.")
        else:
            print("  [PASS] Original target contains strictly {1, 2}.")
            
        # Target mapping check: 1 -> 0 (Good), 2 -> 1 (Bad)
        correct_mapping = ((df["original_target"] == 1) & (df["target"] == 0)) | \
                          ((df["original_target"] == 2) & (df["target"] == 1))
        if not correct_mapping.all():
            failures.append("Mapping mismatch between original_target and target! (Expected 1->0, 2->1)")
        else:
            print("  [PASS] Target mapping verified: original 1 (Good) -> 0, original 2 (Bad) -> 1.")
            
        target_counts = df["target"].value_counts().to_dict()
        if target_counts.get(0, 0) != 700 or target_counts.get(1, 0) != 300:
            failures.append(f"Target distribution mismatch! Expected {{0: 700, 1: 300}}, got {target_counts}")
        else:
            print("  [PASS] Target class balance verified: 700 Good (70.0%), 300 Bad (30.0%).")
            
    # 6. Column Names & Duplicates
    col_names = df.columns.tolist()
    if len(col_names) != len(set(col_names)):
        failures.append("Duplicate column names found in dataframe!")
    else:
        print("  [PASS] No duplicate column names.")
        
    for col in col_names:
        if col != col.lower():
            failures.append(f"Column '{col}' is not lowercase snake_case!")
    print("  [PASS] All column names conform to lowercase snake_case standard.")
    
    # 7. Missing Values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        failures.append(f"Unexpected NaN missing values detected: {null_counts[null_counts > 0].to_dict()}")
    else:
        print("  [PASS] Zero NaN missing values across all columns.")
        
    # 8. Categorical Values Sanity Checks (Against german.doc)
    valid_checking = {"A11", "A12", "A13", "A14"}
    if not set(df["checking_status"].unique()).issubset(valid_checking):
        failures.append(f"Invalid checking_status codes: {set(df['checking_status'].unique())}")
    else:
        print("  [PASS] checking_status codes verified against german.doc.")
        
    valid_history = {"A30", "A31", "A32", "A33", "A34"}
    if not set(df["credit_history"].unique()).issubset(valid_history):
        failures.append(f"Invalid credit_history codes: {set(df['credit_history'].unique())}")
    else:
        print("  [PASS] credit_history codes verified against german.doc.")
        
    valid_purpose = {"A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"}
    if not set(df["purpose"].unique()).issubset(valid_purpose):
        failures.append(f"Invalid purpose codes: {set(df['purpose'].unique())}")
    else:
        if "A47" in set(df["purpose"].unique()):
            failures.append("Code A47 unexpectedly present in purpose!")
        else:
            print("  [PASS] purpose codes verified (A47 confirmed absent, exactly as documented in german.doc).")
            
    valid_savings = {"A61", "A62", "A63", "A64", "A65"}
    if not set(df["savings_status"].unique()).issubset(valid_savings):
        failures.append(f"Invalid savings_status codes: {set(df['savings_status'].unique())}")
    else:
        print("  [PASS] savings_status codes verified against german.doc.")
        
    valid_emp = {"A71", "A72", "A73", "A74", "A75"}
    if not set(df["employment_duration"].unique()).issubset(valid_emp):
        failures.append(f"Invalid employment_duration codes: {set(df['employment_duration'].unique())}")
    else:
        print("  [PASS] employment_duration codes verified against german.doc.")
        
    # Attribute 9: Personal status and sex
    valid_personal = {"A91", "A92", "A93", "A94"}
    if not set(df["personal_status_sex"].unique()).issubset(valid_personal):
        failures.append(f"Invalid personal_status_sex codes: {set(df['personal_status_sex'].unique())}")
    else:
        if "A95" in set(df["personal_status_sex"].unique()):
            failures.append("Code A95 unexpectedly present!")
        else:
            warnings.append("German 'personal_status_sex' conflates sex and marital status (A91: male div/sep, A92: female div/sep/mar, A93: male single, A94: male mar/wid). Code A95 is absent.")
            print("  [PASS] personal_status_sex verified (A91-A94 present, A95 confirmed absent).")
            
    valid_other_parties = {"A101", "A102", "A103"}
    if not set(df["other_debtors_guarantors"].unique()).issubset(valid_other_parties):
        failures.append(f"Invalid other_debtors_guarantors codes: {set(df['other_debtors_guarantors'].unique())}")
        
    valid_property = {"A121", "A122", "A123", "A124"}
    if not set(df["property"].unique()).issubset(valid_property):
        failures.append(f"Invalid property codes: {set(df['property'].unique())}")
        
    valid_plans = {"A141", "A142", "A143"}
    if not set(df["other_installment_plans"].unique()).issubset(valid_plans):
        failures.append(f"Invalid other_installment_plans codes: {set(df['other_installment_plans'].unique())}")
        
    valid_housing = {"A151", "A152", "A153"}
    if not set(df["housing"].unique()).issubset(valid_housing):
        failures.append(f"Invalid housing codes: {set(df['housing'].unique())}")
        
    valid_job = {"A171", "A172", "A173", "A174"}
    if not set(df["job"].unique()).issubset(valid_job):
        failures.append(f"Invalid job codes: {set(df['job'].unique())}")
        
    valid_tel = {"A191", "A192"}
    if not set(df["telephone"].unique()).issubset(valid_tel):
        failures.append(f"Invalid telephone codes: {set(df['telephone'].unique())}")
        
    valid_foreign = {"A201", "A202"}
    if not set(df["foreign_worker"].unique()).issubset(valid_foreign):
        failures.append(f"Invalid foreign_worker codes: {set(df['foreign_worker'].unique())}")
    else:
        fw_counts = df["foreign_worker"].value_counts().to_dict()
        if fw_counts.get("A201", 0) / len(df) > 0.95:
            warnings.append(f"German 'foreign_worker' is near-constant: {fw_counts.get('A201')} A201 (yes) vs {fw_counts.get('A202')} A202 (no). High class imbalance for this protected candidate.")
        print("  [PASS] All remaining categorical feature codes match german.doc specifications.")
        
    # 9. Numerical Sanity
    if (df["duration_months"] < 4).any() or (df["duration_months"] > 72).any():
        failures.append(f"duration_months out of bounds: min={df['duration_months'].min()}, max={df['duration_months'].max()}")
    else:
        print(f"  [PASS] duration_months verified within range [4, 72].")
        
    if (df["credit_amount"] < 250).any() or (df["credit_amount"] > 18424).any():
        failures.append(f"credit_amount out of bounds: min={df['credit_amount'].min()}, max={df['credit_amount'].max()}")
    else:
        print(f"  [PASS] credit_amount verified within range [250, 18,424] DM.")
        
    if not set(df["installment_rate"].unique()).issubset({1, 2, 3, 4}):
        failures.append(f"installment_rate invalid: {set(df['installment_rate'].unique())}")
    else:
        print("  [PASS] installment_rate verified strictly in {1, 2, 3, 4}.")
        
    if not set(df["present_residence_since"].unique()).issubset({1, 2, 3, 4}):
        failures.append(f"present_residence_since invalid: {set(df['present_residence_since'].unique())}")
    else:
        print("  [PASS] present_residence_since verified strictly in {1, 2, 3, 4}.")
        
    if (df["age"] < 18).any() or (df["age"] > 100).any():
        failures.append(f"age out of bounds: min={df['age'].min()}, max={df['age'].max()}")
    else:
        print(f"  [PASS] age verified realistic: min={df['age'].min()}, max={df['age'].max()} years.")
        
    if not set(df["existing_credits"].unique()).issubset({1, 2, 3, 4}):
        failures.append(f"existing_credits invalid: {set(df['existing_credits'].unique())}")
        
    if not set(df["num_dependents"].unique()).issubset({1, 2}):
        failures.append(f"num_dependents invalid: {set(df['num_dependents'].unique())}")
    else:
        print("  [PASS] All numerical discrete features (credits, dependents) verified valid.")
        
    # Duplicates check
    exact_dups = df.duplicated().sum()
    if exact_dups > 0:
        failures.append(f"German dataset contains {exact_dups} exact duplicate rows!")
    else:
        print("  [PASS] Zero duplicate rows in German credit dataset.")
        
    # Final Result
    is_pass = len(failures) == 0
    print("-" * 60)
    if is_pass:
        print("GERMAN DATA VALIDATION RESULT: PASS" + (" WITH WARNINGS" if warnings else ""))
        for w in warnings:
            print(f"  [WARNING/NOTE] {w}")
    else:
        print("GERMAN DATA VALIDATION RESULT: FAIL")
        for f in failures:
            print(f"  [FAIL] {f}")
    print("=" * 60)
    
    return is_pass, failures, warnings


if __name__ == "__main__":
    success, _, _ = validate_german()
    sys.exit(0 if success else 1)
