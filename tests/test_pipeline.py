"""
Basic tests for the loan approval pipeline.

Run with:
    pytest

These are intentionally simple "does it run and does it make sense"
tests, appropriate for a portfolio/learning project - they check that
training produces a usable model and that predictions come back in the
expected shape and range, not exhaustive ML-quality testing.
"""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from predict import LoanApprovalModel  # noqa: E402

MODEL_PATH = ROOT / "models" / "loan_model.pkl"


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("models/loan_model.pkl not found - run `python src/train.py` first.")
    return LoanApprovalModel()


SAMPLE_APPLICANT = {
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


def test_model_loads(model):
    assert model.pipeline is not None
    assert model.model_name in {"logistic_regression", "random_forest"}


def test_predict_returns_expected_shape(model):
    result = model.predict(SAMPLE_APPLICANT)
    assert set(result.keys()) == {"approved", "approval_probability"}
    assert isinstance(result["approved"], bool)
    assert 0.0 <= result["approval_probability"] <= 1.0


def test_predict_handles_missing_field(model):
    """The pipeline's own imputer should handle a missing field gracefully."""
    applicant = dict(SAMPLE_APPLICANT)
    del applicant["Savings"]
    result = model.predict(applicant)
    assert 0.0 <= result["approval_probability"] <= 1.0


def test_higher_credit_score_does_not_lower_approval_odds():
    """
    Sanity/regression check: all else equal, a much better credit score
    should not make the predicted approval probability go down.
    Guards against accidental label-flipping bugs (e.g. Yes/No mixed up).
    """
    if not MODEL_PATH.exists():
        pytest.skip("models/loan_model.pkl not found - run `python src/train.py` first.")
    m = LoanApprovalModel()

    low_score = dict(SAMPLE_APPLICANT, Credit_Score=450)
    high_score = dict(SAMPLE_APPLICANT, Credit_Score=820)

    low_result = m.predict(low_score)
    high_result = m.predict(high_score)

    assert high_result["approval_probability"] >= low_result["approval_probability"]
