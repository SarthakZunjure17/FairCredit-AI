"""
Data Ingestion, Auditing, and Standardization Script for Responsible AI Research
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Author: Lead ML Data Engineer
Philosophy: "Garbage data -> garbage ML. The data pipeline must be correct, reproducible, documented, and auditable."
"""

import os
import json
import numpy as np
import pandas as pd


def get_project_root():
    """Return the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def profile_dataframe(df, dataset_name, role_map, sensitive_notes=None):
    """
    Generate comprehensive statistical profile of a dataframe.
    Calculates 20 specific audit dimensions required by the research protocol.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    col_profiles = []
    
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        unique_vals = s.nunique(dropna=False)
        missing_count = int(s.isnull().sum())
        missing_pct = float(round((missing_count / total_rows) * 100, 4))
        
        is_num = pd.api.types.is_numeric_dtype(s)
        
        # Calculate statistics
        if is_num:
            min_val = float(s.min()) if not s.isnull().all() else None
            max_val = float(s.max()) if not s.isnull().all() else None
            mean_val = float(round(s.mean(), 4)) if not s.isnull().all() else None
            median_val = float(round(s.median(), 4)) if not s.isnull().all() else None
            std_val = float(round(s.std(), 4)) if not s.isnull().all() else None
        else:
            min_val, max_val, mean_val, median_val, std_val = None, None, None, None, None
            
        # Frequency table for categories / discrete values
        val_counts = s.value_counts(dropna=False)
        if unique_vals <= 25:
            # Full frequencies for low-cardinality
            freq_dict = {str(k): int(v) for k, v in val_counts.items()}
        else:
            # Top 5 and bottom 5 for high cardinality
            top5 = val_counts.head(5).to_dict()
            freq_dict = {f"top_{k}": int(v) for k, v in top5.items()}
            freq_dict["unique_count_total"] = int(unique_vals)
            
        # Constant and near-constant checks
        is_constant = bool(unique_vals == 1)
        top_freq = val_counts.iloc[0] if len(val_counts) > 0 else 0
        is_near_constant = bool(top_freq / total_rows >= 0.95 and not is_constant)
        
        # Role assignment
        role = role_map.get(col, "feature")
        
        # Suspicious / impossible value notes
        suspicious_notes = []
        if is_num and (s < 0).any():
            if "bill_amt" in col:
                suspicious_notes.append("Contains negative values: legitimate credit balance (overpayment/refund)")
            elif "pay" in col and "amt" not in col:
                suspicious_notes.append("Contains -2 and -1: represents inactive and paid-in-full statuses")
            else:
                suspicious_notes.append("Contains negative numerical values")
        if is_near_constant:
            suspicious_notes.append(f"Near-constant column: {round((top_freq / total_rows)*100, 2)}% single value")
        if sensitive_notes and col in sensitive_notes:
            suspicious_notes.append(sensitive_notes[col])
            
        col_profile = {
            "column_name": col,
            "role": role,
            "data_type": dtype_str,
            "unique_count": int(unique_vals),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "is_constant": is_constant,
            "is_near_constant": is_near_constant,
            "frequencies": freq_dict,
            "suspicious_notes": "; ".join(suspicious_notes) if suspicious_notes else "None"
        }
        col_profiles.append(col_profile)
        
    # Dataset level summary
    exact_duplicates = int(df.duplicated().sum())
    
    # Check duplicate IDs if ID column exists
    id_col = [c for c, r in role_map.items() if r == "identifier"]
    if id_col:
        dup_ids = int(df[id_col[0]].duplicated().sum())
        feature_cols = [c for c in df.columns if c != id_col[0]]
        feature_duplicates = int(df[feature_cols].duplicated().sum())
    else:
        dup_ids = 0
        feature_duplicates = exact_duplicates
        
    summary = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_ids": dup_ids,
        "feature_duplicate_rows_excluding_id": feature_duplicates,
        "target_distribution": {
            str(k): int(v) for k, v in df["target"].value_counts().items()
        } if "target" in df.columns else {},
        "target_percentage": {
            str(k): float(round((v / total_rows) * 100, 2)) for k, v in df["target"].value_counts().items()
        } if "target" in df.columns else {},
        "columns": col_profiles
    }
    
    return summary


def run_pipeline():
    project_root = get_project_root()
    print("=" * 70)
    print("STARTING RESPONSIBLE AI DATA AUDIT & STANDARDIZATION PIPELINE")
    print(f"Project Root: {project_root}")
    print("=" * 70)
    
    # Directories
    raw_taiwan_path = os.path.join(project_root, "data", "raw", "taiwan", "default of credit card clients.xls")
    raw_german_path = os.path.join(project_root, "data", "raw", "german", "german.data")
    
    processed_dir = os.path.join(project_root, "data", "processed")
    metadata_dir = os.path.join(project_root, "metadata")
    reports_dir = os.path.join(project_root, "reports")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Verify raw files exist
    if not os.path.exists(raw_taiwan_path):
        raise FileNotFoundError(f"Raw Taiwan dataset not found at {raw_taiwan_path}")
    if not os.path.exists(raw_german_path):
        raise FileNotFoundError(f"Raw German dataset not found at {raw_german_path}")
        
    print("\n[STEP 1] Ingesting and Auditing Taiwan Credit Dataset...")
    # Inspect raw structure: Row 0 has X1-X23, Row 1 has column names, Rows 2-30001 are data rows
    df_taiwan_raw = pd.read_excel(raw_taiwan_path, header=1)
    print(f"Loaded Taiwan raw data. Shape: {df_taiwan_raw.shape}")
    
    # Column mapping Taiwan
    taiwan_col_map = {
        "ID": "id",
        "LIMIT_BAL": "limit_bal",
        "SEX": "sex",
        "EDUCATION": "education",
        "MARRIAGE": "marriage",
        "AGE": "age",
        "PAY_0": "pay_0",
        "PAY_2": "pay_2",
        "PAY_3": "pay_3",
        "PAY_4": "pay_4",
        "PAY_5": "pay_5",
        "PAY_6": "pay_6",
        "BILL_AMT1": "bill_amt1",
        "BILL_AMT2": "bill_amt2",
        "BILL_AMT3": "bill_amt3",
        "BILL_AMT4": "bill_amt4",
        "BILL_AMT5": "bill_amt5",
        "BILL_AMT6": "bill_amt6",
        "PAY_AMT1": "pay_amt1",
        "PAY_AMT2": "pay_amt2",
        "PAY_AMT3": "pay_amt3",
        "PAY_AMT4": "pay_amt4",
        "PAY_AMT5": "pay_amt5",
        "PAY_AMT6": "pay_amt6",
        "default payment next month": "original_target"
    }
    
    taiwan_roles = {
        "id": "identifier",
        "limit_bal": "feature",
        "sex": "candidate_protected_attribute",
        "education": "feature",
        "marriage": "candidate_protected_attribute",
        "age": "candidate_protected_attribute",
        "pay_0": "feature",
        "pay_2": "feature",
        "pay_3": "feature",
        "pay_4": "feature",
        "pay_5": "feature",
        "pay_6": "feature",
        "bill_amt1": "feature",
        "bill_amt2": "feature",
        "bill_amt3": "feature",
        "bill_amt4": "feature",
        "bill_amt5": "feature",
        "bill_amt6": "feature",
        "pay_amt1": "feature",
        "pay_amt2": "feature",
        "pay_amt3": "feature",
        "pay_amt4": "feature",
        "pay_amt5": "feature",
        "pay_amt6": "feature",
        "original_target": "target",
        "target": "target"
    }
    
    taiwan_sensitive_notes = {
        "sex": "Protected attribute: 1=Male (24.17% default), 2=Female (20.78% default)",
        "marriage": "Candidate protected attribute under ECOA: 1=Married, 2=Single, 3=Others, 0=Undocumented",
        "age": "Candidate protected attribute: continuous (21-79). Disparities exist between young (<25) and prime age"
    }
    
    df_taiwan = df_taiwan_raw.rename(columns=taiwan_col_map)
    # Add standardized target column: 0=non-default, 1=default (preserving original_target)
    df_taiwan["target"] = df_taiwan["original_target"].astype(int)
    
    # Convert data types safely
    int_cols_taiwan = [
        "id", "sex", "education", "marriage", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "original_target", "target"
    ]
    float_cols_taiwan = [
        "limit_bal",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"
    ]
    for c in int_cols_taiwan:
        df_taiwan[c] = df_taiwan[c].astype(int)
    for c in float_cols_taiwan:
        df_taiwan[c] = df_taiwan[c].astype(float)
        
    print(f"Taiwan standardized dataframe shape: {df_taiwan.shape}")
    print(f"Taiwan target distribution: {df_taiwan['target'].value_counts().to_dict()}")
    
    # Save canonical Taiwan datasets
    taiwan_clean_csv = os.path.join(processed_dir, "taiwan_credit_clean.csv")
    taiwan_clean_parquet = os.path.join(processed_dir, "taiwan_credit_clean.parquet")
    df_taiwan.to_csv(taiwan_clean_csv, index=False)
    df_taiwan.to_parquet(taiwan_clean_parquet, index=False)
    print(f"Saved canonical Taiwan datasets to:\n  {taiwan_clean_csv}\n  {taiwan_clean_parquet}")
    
    # Profile Taiwan
    taiwan_profile = profile_dataframe(df_taiwan, "Taiwan Credit Card Default", taiwan_roles, taiwan_sensitive_notes)
    with open(os.path.join(reports_dir, "taiwan_profile.json"), "w", encoding="utf-8") as f:
        json.dump(taiwan_profile, f, indent=2)
    # Also save CSV summary of profile
    df_taiwan_profile = pd.DataFrame(taiwan_profile["columns"]).drop(columns=["frequencies"])
    df_taiwan_profile.to_csv(os.path.join(reports_dir, "taiwan_profile.csv"), index=False)
    print("Generated Taiwan profile JSON and CSV.")
    
    print("\n[STEP 2] Ingesting and Auditing Statlog German Credit Dataset...")
    # German raw has 21 space-separated columns, no header
    german_raw_cols = [
        "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
        "savings_status", "employment_duration", "installment_rate", "personal_status_sex",
        "other_debtors_guarantors", "present_residence_since", "property", "age",
        "other_installment_plans", "housing", "existing_credits", "job", "num_dependents",
        "telephone", "foreign_worker", "original_target"
    ]
    
    df_german_raw = pd.read_csv(raw_german_path, sep=" ", header=None, names=german_raw_cols)
    print(f"Loaded German raw data. Shape: {df_german_raw.shape}")
    
    # Standardize target:
    # Original german.doc: 1 = Good, 2 = Bad
    # Standardized research convention: 0 = Good / Non-Default, 1 = Bad / Default
    df_german = df_german_raw.copy()
    df_german["target"] = (df_german["original_target"] == 2).astype(int)
    
    german_roles = {
        "checking_status": "feature",
        "duration_months": "feature",
        "credit_history": "feature",
        "purpose": "feature",
        "credit_amount": "feature",
        "savings_status": "feature",
        "employment_duration": "feature",
        "installment_rate": "feature",
        "personal_status_sex": "candidate_protected_attribute",
        "other_debtors_guarantors": "feature",
        "present_residence_since": "feature",
        "property": "feature",
        "age": "candidate_protected_attribute",
        "other_installment_plans": "feature",
        "housing": "feature",
        "existing_credits": "feature",
        "job": "feature",
        "num_dependents": "feature",
        "telephone": "feature",
        "foreign_worker": "candidate_protected_attribute",
        "original_target": "target",
        "target": "target"
    }
    
    german_sensitive_notes = {
        "personal_status_sex": "Conflated sex & marital status. A92=Female (35.16% Bad), A91/A93/A94=Male (27.68% Bad)",
        "age": "Candidate protected attribute: continuous (19-75). Young applicants (<=25) have 42.11% bad credit rate vs 27.16% for >25",
        "foreign_worker": "Candidate protected attribute: A201=Foreign worker (96.3%, 30.74% Bad), A202=Non-foreign (3.7%, 10.81% Bad)"
    }
    
    # Save canonical German datasets
    german_clean_csv = os.path.join(processed_dir, "german_credit_clean.csv")
    german_clean_parquet = os.path.join(processed_dir, "german_credit_clean.parquet")
    df_german.to_csv(german_clean_csv, index=False)
    df_german.to_parquet(german_clean_parquet, index=False)
    print(f"Saved canonical German datasets to:\n  {german_clean_csv}\n  {german_clean_parquet}")
    
    # Profile German
    german_profile = profile_dataframe(df_german, "Statlog German Credit", german_roles, german_sensitive_notes)
    with open(os.path.join(reports_dir, "german_profile.json"), "w", encoding="utf-8") as f:
        json.dump(german_profile, f, indent=2)
    df_german_profile = pd.DataFrame(german_profile["columns"]).drop(columns=["frequencies"])
    df_german_profile.to_csv(os.path.join(reports_dir, "german_profile.csv"), index=False)
    print("Generated German profile JSON and CSV.")
    
    print("\n[STEP 3] Generating Column Mappings & Metadata...")
    # Column mapping Taiwan
    taiwan_mapping_records = [
        {"original_column": "ID", "uci_code": "None", "standardized_column": "id", "role": "identifier", "dtype": "int64", "description": "Unique client identifier (1 to 30000)"},
        {"original_column": "LIMIT_BAL", "uci_code": "X1", "standardized_column": "limit_bal", "role": "feature", "dtype": "float64", "description": "Amount of given credit in NT dollars (individual and family credit)"},
        {"original_column": "SEX", "uci_code": "X2", "standardized_column": "sex", "role": "candidate_protected_attribute", "dtype": "int64", "description": "Gender: 1=Male, 2=Female"},
        {"original_column": "EDUCATION", "uci_code": "X3", "standardized_column": "education", "role": "feature", "dtype": "int64", "description": "Education level: 1=Graduate school, 2=University, 3=High school, 4=Others, 0/5/6=Undocumented"},
        {"original_column": "MARRIAGE", "uci_code": "X4", "standardized_column": "marriage", "role": "candidate_protected_attribute", "dtype": "int64", "description": "Marital status: 1=Married, 2=Single, 3=Others, 0=Undocumented"},
        {"original_column": "AGE", "uci_code": "X5", "standardized_column": "age", "role": "candidate_protected_attribute", "dtype": "int64", "description": "Age in years (21 to 79)"},
        {"original_column": "PAY_0", "uci_code": "X6", "standardized_column": "pay_0", "role": "feature", "dtype": "int64", "description": "Repayment status in September 2005 (-2=inactive, -1=paid duly, 0=revolving, 1-8=months delayed)"},
        {"original_column": "PAY_2", "uci_code": "X7", "standardized_column": "pay_2", "role": "feature", "dtype": "int64", "description": "Repayment status in August 2005 (-2=inactive, -1=paid duly, 0=revolving, 1-8=months delayed)"},
        {"original_column": "PAY_3", "uci_code": "X8", "standardized_column": "pay_3", "role": "feature", "dtype": "int64", "description": "Repayment status in July 2005 (-2=inactive, -1=paid duly, 0=revolving, 1-8=months delayed)"},
        {"original_column": "PAY_4", "uci_code": "X9", "standardized_column": "pay_4", "role": "feature", "dtype": "int64", "description": "Repayment status in June 2005 (-2=inactive, -1=paid duly, 0=revolving, 1-8=months delayed)"},
        {"original_column": "PAY_5", "uci_code": "X10", "standardized_column": "pay_5", "role": "feature", "dtype": "int64", "description": "Repayment status in May 2005 (-2=inactive, -1=paid duly, 0=revolving, 2-8=months delayed)"},
        {"original_column": "PAY_6", "uci_code": "X11", "standardized_column": "pay_6", "role": "feature", "dtype": "int64", "description": "Repayment status in April 2005 (-2=inactive, -1=paid duly, 0=revolving, 2-8=months delayed)"},
        {"original_column": "BILL_AMT1", "uci_code": "X12", "standardized_column": "bill_amt1", "role": "feature", "dtype": "float64", "description": "Amount of bill statement in September 2005 (NT dollar)"},
        {"original_column": "BILL_AMT2", "uci_code": "X13", "standardized_column": "bill_amt2", "role": "feature", "dtype": "float64", "description": "Amount of bill statement in August 2005 (NT dollar)"},
        {"original_column": "BILL_AMT3", "uci_code": "X14", "standardized_column": "bill_amt3", "role": "feature", "dtype": "float64", "description": "Amount of bill statement in July 2005 (NT dollar)"},
        {"original_column": "BILL_AMT4", "uci_code": "X15", "standardized_column": "bill_amt4", "role": "feature", "dtype": "float64", "description": "Amount of bill statement in June 2005 (NT dollar)"},
        {"original_column": "BILL_AMT5", "uci_code": "X16", "standardized_column": "bill_amt5", "role": "feature", "dtype": "float64", "description": "Amount of bill statement in May 2005 (NT dollar)"},
        {"original_column": "BILL_AMT6", "uci_code": "X17", "standardized_column": "bill_amt6", "role": "feature", "dtype": "float64", "description": "Amount of bill statement in April 2005 (NT dollar)"},
        {"original_column": "PAY_AMT1", "uci_code": "X18", "standardized_column": "pay_amt1", "role": "feature", "dtype": "float64", "description": "Amount of previous payment in September 2005 (NT dollar)"},
        {"original_column": "PAY_AMT2", "uci_code": "X19", "standardized_column": "pay_amt2", "role": "feature", "dtype": "float64", "description": "Amount of previous payment in August 2005 (NT dollar)"},
        {"original_column": "PAY_AMT3", "uci_code": "X20", "standardized_column": "pay_amt3", "role": "feature", "dtype": "float64", "description": "Amount of previous payment in July 2005 (NT dollar)"},
        {"original_column": "PAY_AMT4", "uci_code": "X21", "standardized_column": "pay_amt4", "role": "feature", "dtype": "float64", "description": "Amount of previous payment in June 2005 (NT dollar)"},
        {"original_column": "PAY_AMT5", "uci_code": "X22", "standardized_column": "pay_amt5", "role": "feature", "dtype": "float64", "description": "Amount of previous payment in May 2005 (NT dollar)"},
        {"original_column": "PAY_AMT6", "uci_code": "X23", "standardized_column": "pay_amt6", "role": "feature", "dtype": "float64", "description": "Amount of previous payment in April 2005 (NT dollar)"},
        {"original_column": "default payment next month", "uci_code": "Y", "standardized_column": "original_target", "role": "target", "dtype": "int64", "description": "Original default indicator: 1=default in October 2005, 0=non-default"},
        {"original_column": "default payment next month", "uci_code": "Y", "standardized_column": "target", "role": "target", "dtype": "int64", "description": "Standardized binary target: 1=Credit Default / Bad Risk, 0=No Default / Good Credit"}
    ]
    pd.DataFrame(taiwan_mapping_records).to_csv(os.path.join(metadata_dir, "column_mapping_taiwan.csv"), index=False)
    
    # Column mapping German
    german_mapping_records = [
        {"attribute_index": "Attribute 1", "raw_doc_name": "Status of existing checking account", "standardized_column": "checking_status", "role": "feature", "dtype": "object", "description": "Checking account status: A11 (<0 DM), A12 (0-200 DM), A13 (>=200 DM / salary), A14 (no checking)"},
        {"attribute_index": "Attribute 2", "raw_doc_name": "Duration in month", "standardized_column": "duration_months", "role": "feature", "dtype": "int64", "description": "Duration of credit in months (4 to 72)"},
        {"attribute_index": "Attribute 3", "raw_doc_name": "Credit history", "standardized_column": "credit_history", "role": "feature", "dtype": "object", "description": "Credit history: A30 (no credits/paid duly), A31 (all at this bank paid duly), A32 (existing paid duly), A33 (delay in past), A34 (critical account/other credits)"},
        {"attribute_index": "Attribute 4", "raw_doc_name": "Purpose", "standardized_column": "purpose", "role": "feature", "dtype": "object", "description": "Loan purpose: A40 (new car), A41 (used car), A42 (furniture), A43 (radio/tv), A44 (domestic appliances), A45 (repairs), A46 (education), A48 (retraining), A49 (business), A410 (others)"},
        {"attribute_index": "Attribute 5", "raw_doc_name": "Credit amount", "standardized_column": "credit_amount", "role": "feature", "dtype": "int64", "description": "Credit amount in DM (250 to 18424)"},
        {"attribute_index": "Attribute 6", "raw_doc_name": "Savings account/bonds", "standardized_column": "savings_status", "role": "feature", "dtype": "object", "description": "Savings/bonds balance: A61 (<100 DM), A62 (100-500 DM), A63 (500-1000 DM), A64 (>=1000 DM), A65 (unknown/no savings)"},
        {"attribute_index": "Attribute 7", "raw_doc_name": "Present employment since", "standardized_column": "employment_duration", "role": "feature", "dtype": "object", "description": "Present employment duration: A71 (unemployed), A72 (<1 yr), A73 (1-4 yrs), A74 (4-7 yrs), A75 (>=7 yrs)"},
        {"attribute_index": "Attribute 8", "raw_doc_name": "Installment rate in percentage of disposable income", "standardized_column": "installment_rate", "role": "feature", "dtype": "int64", "description": "Installment rate percentage: 1, 2, 3, 4"},
        {"attribute_index": "Attribute 9", "raw_doc_name": "Personal status and sex", "standardized_column": "personal_status_sex", "role": "candidate_protected_attribute", "dtype": "object", "description": "Sex and marital status: A91 (male: divorced/sep), A92 (female: div/sep/mar), A93 (male: single), A94 (male: mar/wid)"},
        {"attribute_index": "Attribute 10", "raw_doc_name": "Other debtors / guarantors", "standardized_column": "other_debtors_guarantors", "role": "feature", "dtype": "object", "description": "Other parties: A101 (none), A102 (co-applicant), A103 (guarantor)"},
        {"attribute_index": "Attribute 11", "raw_doc_name": "Present residence since", "standardized_column": "present_residence_since", "role": "feature", "dtype": "int64", "description": "Years at present residence: 1, 2, 3, 4"},
        {"attribute_index": "Attribute 12", "raw_doc_name": "Property", "standardized_column": "property", "role": "feature", "dtype": "object", "description": "Most valuable property: A121 (real estate), A122 (building society/life ins), A123 (car/other), A124 (unknown/no property)"},
        {"attribute_index": "Attribute 13", "raw_doc_name": "Age in years", "standardized_column": "age", "role": "candidate_protected_attribute", "dtype": "int64", "description": "Age in years (19 to 75)"},
        {"attribute_index": "Attribute 14", "raw_doc_name": "Other installment plans", "standardized_column": "other_installment_plans", "role": "feature", "dtype": "object", "description": "Other installment plans: A141 (bank), A142 (stores), A143 (none)"},
        {"attribute_index": "Attribute 15", "raw_doc_name": "Housing", "standardized_column": "housing", "role": "feature", "dtype": "object", "description": "Housing status: A151 (rent), A152 (own), A153 (for free)"},
        {"attribute_index": "Attribute 16", "raw_doc_name": "Number of existing credits at this bank", "standardized_column": "existing_credits", "role": "feature", "dtype": "int64", "description": "Number of existing credits: 1, 2, 3, 4"},
        {"attribute_index": "Attribute 17", "raw_doc_name": "Job", "standardized_column": "job", "role": "feature", "dtype": "object", "description": "Job qualification: A171 (unemployed/unskilled non-res), A172 (unskilled res), A173 (skilled/official), A174 (management/self-emp/highly qual)"},
        {"attribute_index": "Attribute 18", "raw_doc_name": "Number of people being liable to provide maintenance for", "standardized_column": "num_dependents", "role": "feature", "dtype": "int64", "description": "Number of maintenance dependents: 1, 2"},
        {"attribute_index": "Attribute 19", "raw_doc_name": "Telephone", "standardized_column": "telephone", "role": "feature", "dtype": "object", "description": "Telephone registration: A191 (none), A192 (yes, registered customer name)"},
        {"attribute_index": "Attribute 20", "raw_doc_name": "foreign worker", "standardized_column": "foreign_worker", "role": "candidate_protected_attribute", "dtype": "object", "description": "Foreign worker status: A201 (yes), A202 (no)"},
        {"attribute_index": "Target/Class", "raw_doc_name": "Cost Matrix / Class", "standardized_column": "original_target", "role": "target", "dtype": "int64", "description": "Original raw class label: 1=Good, 2=Bad"},
        {"attribute_index": "Target/Class", "raw_doc_name": "Standardized Target", "standardized_column": "target", "role": "target", "dtype": "int64", "description": "Standardized binary target: 1=Bad Credit / Risk Event, 0=Good Credit"}
    ]
    pd.DataFrame(german_mapping_records).to_csv(os.path.join(metadata_dir, "column_mapping_german.csv"), index=False)
    print("Saved column mappings to metadata directory.")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Taiwan Canonical Dataset: {taiwan_clean_csv} ({len(df_taiwan)} rows, {len(df_taiwan.columns)} columns)")
    print(f"German Canonical Dataset: {german_clean_csv} ({len(df_german)} rows, {len(df_german.columns)} columns)")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
