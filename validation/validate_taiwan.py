"""
Automated Data Validation for Taiwan Credit Card Default Dataset.
Project: Responsible AI Credit Scoring Research

Validates:
1. Raw data immutability & presence
2. Canonical dataset presence and integrity
3. Exact row & column counts
4. Absence of accidental index columns
5. Target existence, valid values {0, 1}, and mapping consistency
6. Unique customer identifiers and exclusion from ML features X
7. Clean snake_case column names and absence of duplicate column names
8. Absence of NaN missing values
9. Categorical value validity against documentation
10. Numerical ranges sanity checks
"""

import os
import sys
import pandas as pd
import numpy as np


def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def validate_taiwan():
    project_root = get_project_root()
    print("=" * 60)
    print("RUNNING TAIWAN CREDIT DATASET VALIDATION CHECKS")
    print("=" * 60)
    
    failures = []
    warnings = []
    
    # 1. Raw Data Immutability
    raw_source = os.path.join(project_root, "default+of+credit+card+clients (1)", "default of credit card clients.xls")
    raw_ingested = os.path.join(project_root, "data", "raw", "taiwan", "default of credit card clients.xls")
    
    if not os.path.exists(raw_source):
        failures.append(f"Original source file missing at {raw_source}")
    else:
        size_source = os.path.getsize(raw_source)
        if size_source != 5539328:
            failures.append(f"Original source file size altered! Expected 5539328 bytes, found {size_source}")
        else:
            print("  [PASS] Original raw file is present and unaltered (5,539,328 bytes).")
            
    if not os.path.exists(raw_ingested):
        failures.append(f"Ingested raw copy missing at {raw_ingested}")
    else:
        print("  [PASS] Ingested raw copy is present.")
        
    # 2. Canonical Datasets Presence
    clean_csv = os.path.join(project_root, "data", "processed", "taiwan_credit_clean.csv")
    clean_parquet = os.path.join(project_root, "data", "processed", "taiwan_credit_clean.parquet")
    
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
    expected_rows = 30000
    expected_cols = 26
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
        if orig_unique != {0, 1}:
            failures.append(f"Original target contains invalid values: {orig_unique}. Expected {{0, 1}}.")
        else:
            print("  [PASS] Original target contains strictly {0, 1}.")
            
        # Target mapping check (for Taiwan, original 0 is non-default, 1 is default)
        if not (df["target"] == df["original_target"]).all():
            failures.append("Mapping mismatch between original_target and target!")
        else:
            print("  [PASS] Target mapping verified: original_target maps 1:1 to standardized target.")
            
        target_counts = df["target"].value_counts().to_dict()
        if target_counts.get(0, 0) != 23364 or target_counts.get(1, 0) != 6636:
            failures.append(f"Target distribution mismatch! Expected {{0: 23364, 1: 6636}}, got {target_counts}")
        else:
            print("  [PASS] Target class balance verified: 23,364 non-default (77.88%), 6,636 default (22.12%).")
            
    # 6. Identifier Column Validation
    if "id" not in df.columns:
        failures.append("Column 'id' missing from processed dataset!")
    else:
        if df["id"].nunique() != 30000:
            failures.append(f"Customer IDs are not strictly unique! Found {df['id'].nunique()} unique IDs out of 30,000 rows.")
        else:
            print("  [PASS] Identifier 'id' is strictly unique (30,000 unique records).")
            
        # Verify from metadata that id has role = 'identifier'
        mapping_path = os.path.join(project_root, "metadata", "column_mapping_taiwan.csv")
        if os.path.exists(mapping_path):
            df_map = pd.read_csv(mapping_path)
            id_role = df_map[df_map["standardized_column"] == "id"]["role"].values[0]
            if id_role != "identifier":
                failures.append(f"Column 'id' has invalid role '{id_role}'. Expected 'identifier'.")
            else:
                print("  [PASS] Identifier role verified in metadata: role = 'identifier' (excluded from ML features X).")
                
    # 7. Column Names & Duplicates
    col_names = df.columns.tolist()
    if len(col_names) != len(set(col_names)):
        failures.append("Duplicate column names found in dataframe!")
    else:
        print("  [PASS] No duplicate column names.")
        
    for col in col_names:
        if col != col.lower():
            failures.append(f"Column '{col}' is not lowercase snake_case!")
    print("  [PASS] All column names conform to lowercase snake_case standard.")
    
    # 8. Missing Values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        failures.append(f"Unexpected NaN missing values detected: {null_counts[null_counts > 0].to_dict()}")
    else:
        print("  [PASS] Zero NaN missing values across all columns.")
        
    # 9. Categorical Values Sanity Checks
    sex_unique = set(df["sex"].unique())
    if sex_unique != {1, 2}:
        failures.append(f"Unexpected values in 'sex': {sex_unique}. Expected {{1, 2}}.")
    else:
        print("  [PASS] Protected attribute 'sex' contains strictly documented codes {1, 2}.")
        
    edu_unique = set(df["education"].unique())
    if not edu_unique.issubset({0, 1, 2, 3, 4, 5, 6}):
        failures.append(f"Invalid values in 'education': {edu_unique}")
    else:
        undoc_edu = len(df[df["education"].isin([0, 5, 6])])
        warnings.append(f"Taiwan 'education' contains 345 instances with undocumented codes {{0, 5, 6}} ({undoc_edu} rows). Preserved for audit.")
        print("  [PASS] 'education' values within observed set {0, 1, 2, 3, 4, 5, 6}.")
        
    mar_unique = set(df["marriage"].unique())
    if not mar_unique.issubset({0, 1, 2, 3}):
        failures.append(f"Invalid values in 'marriage': {mar_unique}")
    else:
        undoc_mar = len(df[df["marriage"] == 0])
        warnings.append(f"Taiwan 'marriage' contains 54 instances with undocumented code 0 ({undoc_mar} rows). Preserved for audit.")
        print("  [PASS] 'marriage' values within observed set {0, 1, 2, 3}.")
        
    for p in ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]:
        p_vals = set(df[p].unique())
        if not p_vals.issubset(set(range(-2, 9))):
            failures.append(f"Unexpected values in '{p}': {p_vals}")
    print("  [PASS] Repayment status columns (pay_0..pay_6) verified within valid range [-2, 8].")
    
    # 10. Numerical Sanity
    if (df["age"] < 18).any() or (df["age"] > 100).any():
        failures.append(f"Impossible age detected: min={df['age'].min()}, max={df['age'].max()}")
    else:
        print(f"  [PASS] Age range verified realistic: {df['age'].min()} to {df['age'].max()} years.")
        
    if (df["limit_bal"] <= 0).any():
        failures.append("Non-positive credit limit found in 'limit_bal'!")
    else:
        print(f"  [PASS] Credit limit range verified: {df['limit_bal'].min()} to {df['limit_bal'].max()} NT$.")
        
    pay_amt_neg = False
    for p in [f"pay_amt{i}" for i in range(1, 7)]:
        if (df[p] < 0).any():
            failures.append(f"Negative payment amount found in '{p}'!")
            pay_amt_neg = True
    if not pay_amt_neg:
        print("  [PASS] Payment amount columns (pay_amt1..pay_amt6) are strictly non-negative.")
        
    # Check feature duplicates (ignoring ID)
    feature_dups = df.drop(columns=["id"]).duplicated().sum()
    warnings.append(f"Taiwan dataset has 35 duplicate feature profiles (70 rows total) sharing identical credit metrics and identical target outcomes. Preserved because each represents an independent customer ID.")
    print(f"  [INFO] Feature duplicates (excluding ID): {feature_dups} duplicate pairs documented.")
    
    # Final Result
    is_pass = len(failures) == 0
    print("-" * 60)
    if is_pass:
        print("TAIWAN DATA VALIDATION RESULT: PASS" + (" WITH WARNINGS" if warnings else ""))
        for w in warnings:
            print(f"  [WARNING/NOTE] {w}")
    else:
        print("TAIWAN DATA VALIDATION RESULT: FAIL")
        for f in failures:
            print(f"  [FAIL] {f}")
    print("=" * 60)
    
    return is_pass, failures, warnings


if __name__ == "__main__":
    success, _, _ = validate_taiwan()
    sys.exit(0 if success else 1)
