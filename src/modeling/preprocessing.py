"""
Data Ingestion, Splitting, and Preprocessor Builder for Credit Scoring Benchmarks.
Project: Responsible AI Credit Scoring Research

Follows locked research decisions:
1. German Gender: derived sex (A92 -> female, A91/A93/A94 -> male) stored separately.
2. Primary fairness attribute: Taiwan = sex, German = derived sex.
3. Sex excluded from feature matrix X for the primary experiment.
4. Age remains in predictive features X, and age groups are provided as secondary audit attributes.
5. Undocumented category codes in Taiwan (education {0,5,6}, marriage 0) are kept intact.
6. All transformations happen inside sklearn Pipeline/ColumnTransformer fitted strictly on training data.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


def load_dataset(
    dataset_name: str,
    data_dir: str = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, List[str]]]:
    """
    Load canonical dataset and separate features X, target y, and sensitive attributes.

    Args:
        dataset_name: 'taiwan' or 'german' (case-insensitive)
        data_dir: Path to directory containing processed canonical CSV files.

    Returns:
        X: Feature matrix dataframe (demographic sex and identifiers excluded)
        y: Binary target series (1 = Default/Bad risk, 0 = Non-default/Good credit)
        sensitive_df: Dataframe of sensitive attributes for fairness auditing
        feature_types: Dict mapping 'categorical' and 'numerical' to column lists
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root(), "data", "processed")

    name = dataset_name.lower().strip()

    if name == "taiwan":
        file_path = os.path.join(data_dir, "taiwan_credit_clean.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Taiwan canonical dataset not found at {file_path}")

        df = pd.read_csv(file_path)

        # Target variable (1=default, 0=non-default)
        y = df["target"].astype(int)

        # Sensitive attributes dataframe (for fairness auditing only)
        sensitive_df = pd.DataFrame(index=df.index)
        sensitive_df["sex"] = df["sex"].map({1: "male", 2: "female"})
        sensitive_df["sex_code"] = df["sex"].astype(int)
        sensitive_df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 100],
            labels=["<25", "25-35", "35-50", ">50"],
            right=True
        ).astype(str)
        sensitive_df["age_binary"] = np.where(df["age"] > 25, ">25", "<=25")
        sensitive_df["marriage"] = df["marriage"].map({1: "married", 2: "single", 3: "others", 0: "undocumented"})

        # Features X: Exclude id (identifier), sex (primary protected attribute), and targets
        cols_to_exclude = ["id", "sex", "original_target", "target"]
        feature_cols = [c for c in df.columns if c not in cols_to_exclude]
        X = df[feature_cols].copy()

        # Categorical features in X:
        # Per Decision 5: education {0,5,6} and marriage 0 are preserved as categorical
        categorical_cols = [
            "education", "marriage",
            "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"
        ]
        # Numerical continuous features in X:
        numerical_cols = [
            "limit_bal", "age",
            "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
            "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"
        ]

        feature_types = {
            "categorical": categorical_cols,
            "numerical": numerical_cols
        }

        return X, y, sensitive_df, feature_types

    elif name == "german":
        file_path = os.path.join(data_dir, "german_credit_clean.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"German canonical dataset not found at {file_path}")

        df = pd.read_csv(file_path)

        # Target variable (1=bad credit, 0=good credit)
        y = df["target"].astype(int)

        # Sensitive attributes dataframe (for fairness auditing only)
        # Per Locked Decision 1: A92 -> female; A91, A93, A94 -> male
        sensitive_df = pd.DataFrame(index=df.index)
        sensitive_df["sex"] = df["personal_status_sex"].map({
            "A92": "female",
            "A91": "male",
            "A93": "male",
            "A94": "male"
        })
        sensitive_df["age_group"] = np.where(df["age"] > 25, ">25", "<=25")
        sensitive_df["foreign_worker"] = df["foreign_worker"].map({
            "A201": "foreign",
            "A202": "non_foreign"
        })

        # Features X: Exclude personal_status_sex (encodes sex) and targets
        cols_to_exclude = ["personal_status_sex", "original_target", "target"]
        feature_cols = [c for c in df.columns if c not in cols_to_exclude]
        X = df[feature_cols].copy()

        categorical_cols = [
            "checking_status", "credit_history", "purpose", "savings_status",
            "employment_duration", "other_debtors_guarantors", "property",
            "other_installment_plans", "housing", "job", "telephone", "foreign_worker"
        ]
        numerical_cols = [
            "duration_months", "credit_amount", "installment_rate",
            "present_residence_since", "age", "existing_credits", "num_dependents"
        ]

        feature_types = {
            "categorical": categorical_cols,
            "numerical": numerical_cols
        }

        return X, y, sensitive_df, feature_types

    else:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Expected 'taiwan' or 'german'.")


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/test split, ensuring exact index alignment with sensitive attributes.

    Returns:
        X_train, X_test, y_train, y_test, sens_train, sens_test
    """
    # Split indices with stratification on target y
    train_idx, test_idx = train_test_split(
        X.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train = X.loc[train_idx].copy().reset_index(drop=True)
    X_test = X.loc[test_idx].copy().reset_index(drop=True)
    y_train = y.loc[train_idx].copy().reset_index(drop=True)
    y_test = y.loc[test_idx].copy().reset_index(drop=True)
    sens_train = sensitive_df.loc[train_idx].copy().reset_index(drop=True)
    sens_test = sensitive_df.loc[test_idx].copy().reset_index(drop=True)

    return X_train, X_test, y_train, y_test, sens_train, sens_test


def build_preprocessor(
    categorical_cols: List[str],
    numerical_cols: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for preprocessing.
    Categorical columns are one-hot encoded (handle_unknown='ignore').
    Numerical columns are standardized with StandardScaler (if scale_numeric=True) or passed through.

    All transformers are fitted strictly on training data inside the pipeline.
    """
    transformers = []

    # Categorical pipeline
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformers.append(("cat", cat_transformer, categorical_cols))

    # Numerical pipeline
    if scale_numeric:
        num_transformer = StandardScaler()
        transformers.append(("num", num_transformer, numerical_cols))
    else:
        transformers.append(("num", "passthrough", numerical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
