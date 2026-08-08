"""
preprocessing.py
-----------------
Shared data-cleaning and feature-engineering logic used by BOTH the
training script (train.py) and the prediction/app code (predict.py, app.py).

Keeping this logic in one place avoids the classic beginner bug of
preprocessing training data one way and preprocessing "new" data a
different way (which silently breaks predictions).
"""

from __future__ import annotations
import pandas as pd

# Columns dropped because they are just row identifiers, not real features.
ID_COLUMNS = ["Applicant_ID"]

# Target column
TARGET_COLUMN = "Loan_Approved"

# Column that gets label-encoded (binary category)
BINARY_CATEGORICAL_COLUMNS = ["Marital_Status"]

# Columns that get one-hot encoded (multi-class categories)
ONE_HOT_COLUMNS = [
    "Employment_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    return pd.read_csv(csv_path)


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns that carry no predictive signal."""
    cols_present = [c for c in ID_COLUMNS if c in df.columns]
    return df.drop(columns=cols_present)


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (categorical_columns, numerical_columns) present in df."""
    categorical = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical, numerical
