# 🏦 Loan Approval Predictor

A machine learning project that predicts whether a bank loan application
is likely to be **approved** or **rejected**, based on applicant details
such as income, credit score, employment status, and existing debt.

**🔗 Live demo:** _add your deployed Streamlit link here after deploying_

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Demo

<!-- Add a screenshot of the running app here, e.g.: -->
<!-- ![App screenshot](docs/screenshot.png) -->

## Project Overview

This project walks through a complete, realistic ML workflow:

1. **Exploratory Data Analysis (EDA)** – understanding the dataset,
   class balance, and feature distributions (`notebooks/`)
2. **Data cleaning** – handling missing values with imputation
3. **Feature engineering** – one-hot encoding categorical fields, scaling
   numeric fields
4. **Model training** – comparing Logistic Regression vs. Random Forest,
   selected by cross-validated F1 score (`src/train.py`)
5. **Evaluation** – accuracy, precision, recall, F1, ROC-AUC, confusion matrix
6. **Deployment** – an interactive Streamlit web app (`app.py`)

All preprocessing (imputation, encoding, scaling) is bundled into a single
`scikit-learn` `Pipeline`, so the exact transformations used during training
are guaranteed to run identically at prediction time.

## Results

The best model (selected automatically by `train.py`) achieves on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | ~93.7% |
| Precision | ~88.7% |
| Recall | ~91.7% |
| F1 Score | ~90.2% |
| ROC-AUC | ~97.8% |

(Exact numbers are written to `models/metrics.json` every time you retrain.
The dataset is a small, synthetic 1,000-row sample, so these numbers describe
this dataset — not a claim about real-world lending accuracy.)

## Project Structure

```
loan-approval-predictor/
├── app.py                      # Streamlit web app (deployment entry point)
├── data/
│   └── loan_approval_data.csv  # Raw dataset
├── notebooks/
│   └── EDA_and_Model_Training.ipynb   # Exploratory analysis notebook
├── src/
│   ├── preprocessing.py        # Shared data-cleaning helpers
│   ├── train.py                # Trains & saves the model pipeline
│   └── predict.py              # Loads the model & makes predictions
├── models/
│   ├── loan_model.pkl          # Saved trained pipeline (generated)
│   └── metrics.json            # Evaluation results (generated)
├── tests/
│   └── test_pipeline.py        # Basic tests for training/prediction
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/TheYogendraSingh/loan-approval-predictor.git
cd loan-approval-predictor
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

This reads `data/loan_approval_data.csv`, trains and evaluates two models,
and saves the better one to `models/loan_model.pkl`.

### 4. Run the app locally

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

### 5. Run tests

```bash
pytest
```

## Tech Stack

- **Python 3**
- **pandas / numpy** – data handling
- **scikit-learn** – preprocessing pipeline & modeling (Logistic Regression, Random Forest)
- **Streamlit** – web app / deployment
- **matplotlib / seaborn** – exploratory data visualization
- **pytest** – testing

## Dataset

`data/loan_approval_data.csv` — 1,000 synthetic loan applications with
features including applicant/co-applicant income, credit score, employment
status, existing loans, debt-to-income ratio, collateral value, and more.
Target column: `Loan_Approved` (`Yes`/`No`).

## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for the Random Forest
- Try gradient-boosted models (XGBoost / LightGBM)
- Add SHAP-based explainability so applicants can see *why* a decision was made
- Handle class imbalance explicitly (SMOTE / class weights)
- Add CI (GitHub Actions) to auto-run tests on every push

## Author

**Yogendra Singh** — [GitHub](https://github.com/TheYogendraSingh)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
