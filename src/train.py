"""
train.py
--------
End-to-end training script for the Loan Approval Predictor.

Usage:
    python src/train.py

What it does:
    1. Loads data/loan_approval_data.csv
    2. Splits into train/test BEFORE any imputation/encoding/scaling
       (fitting preprocessing on the full dataset before splitting is a
        common beginner mistake called "data leakage" - it lets
        information from the test set leak into training).
    3. Builds a single sklearn Pipeline that bundles:
         - missing-value imputation
         - one-hot encoding of categorical columns
         - feature scaling
         - the classifier itself
       Bundling everything into one Pipeline means the exact same
       transformations are guaranteed to run at prediction time -
       no risk of preprocessing training and new data differently.
    4. Trains two candidate models (Logistic Regression, Random Forest),
       compares them with cross-validation + a held-out test set, and
       saves the better one to models/loan_model.pkl
    5. Saves an evaluation report to models/metrics.json

Run this whenever the dataset changes, to regenerate the model artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import (  # noqa: E402
    TARGET_COLUMN,
    drop_id_columns,
    load_raw_data,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "loan_approval_data.csv"
MODEL_PATH = ROOT / "models" / "loan_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
RANDOM_STATE = 42


def build_pipeline(model, numeric_features, categorical_features) -> Pipeline:
    """Bundle imputation + encoding + scaling + model into ONE pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Make sure loan_approval_data.csv is inside the data/ folder."
        )

    df = load_raw_data(str(DATA_PATH))
    df = drop_id_columns(df)

    # Drop rows where the target itself is missing - we can't train on those.
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    # Encode target: Yes -> 1, No -> 0 (kept simple + explicit, not left to LabelEncoder's
    # alphabetical guess, since alphabetical order for Yes/No happens to work but is
    # a fragile assumption for other label sets).
    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if y.isnull().any():
        raise ValueError(
            "Loan_Approved column contains values other than 'Yes'/'No' - "
            "update the mapping in train.py before continuing."
        )
    X = df.drop(columns=[TARGET_COLUMN])

    # Decide categorical vs numeric by actual dtype rather than a hardcoded
    # list, so this keeps working even if the dataset schema changes slightly.
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = build_pipeline(model, numeric_features, categorical_features)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        pipeline.fit(X_train, y_train)
        test_metrics = evaluate(pipeline, X_test, y_test)
        test_metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        test_metrics["cv_f1_std"] = round(cv_scores.std(), 4)
        results[name] = test_metrics
        fitted_pipelines[name] = pipeline
        print(f"\n[{name}]")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (f1_score={results[best_name]['f1_score']})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "feature_columns": list(X.columns),
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
        },
        MODEL_PATH,
    )
    print(f"Saved best model pipeline to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"Saved metrics report to {METRICS_PATH}")


if __name__ == "__main__":
    main()
