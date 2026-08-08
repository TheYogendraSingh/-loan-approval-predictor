"""
predict.py
----------
Loads the trained pipeline saved by train.py and makes predictions on
new applicant data. Used by app.py (the Streamlit UI) and can also be
run directly for a quick sanity check from the command line.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "loan_model.pkl"


class LoanApprovalModel:
    """Thin wrapper around the saved sklearn Pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python src/train.py` first to create it."
            )
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.model_name = artifact["model_name"]
        self.feature_columns = artifact["feature_columns"]
        self.numeric_features = artifact["numeric_features"]
        self.categorical_features = artifact["categorical_features"]

    def predict(self, applicant: dict) -> dict:
        """
        applicant: dict of {feature_name: value} for ONE applicant.
        Returns: {"approved": bool, "approval_probability": float}
        """
        row = pd.DataFrame([applicant])

        # Make sure every expected column exists (fill missing with NaN so
        # the pipeline's own imputers handle it, instead of crashing).
        for col in self.feature_columns:
            if col not in row.columns:
                row[col] = np.nan
        row = row[self.feature_columns]

        # Ensure dtypes match what the pipeline was fit on - a plain-object
        # column holding a mix of real values and NaN can otherwise confuse
        # sklearn's numeric imputer.
        for col in self.numeric_features:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        for col in self.categorical_features:
            row[col] = row[col].astype(object).where(row[col].notna(), None)

        proba = self.pipeline.predict_proba(row)[0, 1]
        prediction = self.pipeline.predict(row)[0]
        return {
            "approved": bool(prediction == 1),
            "approval_probability": round(float(proba), 4),
        }


if __name__ == "__main__":
    # Quick manual sanity check using values in a typical/plausible range.
    sample_applicant = {
        "Applicant_Income": 65000,
        "Coapplicant_Income": 15000,
        "Employment_Status": "Salaried",
        "Age": 32,
        "Marital_Status": "Married",
        "Dependents": 1,
        "Credit_Score": 720,
        "Existing_Loans": 1,
        "DTI_Ratio": 0.28,
        "Savings": 50000,
        "Collateral_Value": 200000,
        "Loan_Amount": 300000,
        "Loan_Term": 180,
        "Loan_Purpose": "Home",
        "Property_Area": "Urban",
        "Education_Level": "Graduate",
        "Gender": "Male",
        "Employer_Category": "Private",
    }

    model = LoanApprovalModel()
    result = model.predict(sample_applicant)
    print(f"Model used: {model.model_name}")
    print(f"Prediction: {result}")
