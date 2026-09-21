"""
Tests for the trained loan approval model.
"""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/loan_approval_model.joblib"
)

PREPROCESSOR_PATH = Path(
    "models/preprocessor.joblib"
)

DATA_PATH = Path(
    "data/raw/loan_data.csv"
)


NUMERICAL_FEATURES = [
    "Dependents",
    "Applicant_Income",
    "Coapplicant_Income",
    "Loan_Amount",
    "Loan_Term",
    "Credit_History",
    "Age",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Education",
    "Employment_Status",
    "Property_Area",
]


FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


def test_model_file_exists():
    """Check that the final trained model exists."""

    assert MODEL_PATH.exists(), (
        f"Model not found at {MODEL_PATH}"
    )


def test_preprocessor_file_exists():
    """Check that the fitted preprocessor exists."""

    assert PREPROCESSOR_PATH.exists(), (
        f"Preprocessor not found at {PREPROCESSOR_PATH}"
    )


def test_model_can_be_loaded():
    """Check that the trained model can be loaded."""

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_preprocessor_can_be_loaded():
    """Check that the preprocessor can be loaded."""

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    assert preprocessor is not None


def test_model_prediction():
    """Check that the model can generate a prediction."""

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].iloc[[0]].copy()

    X_processed = preprocessor.transform(X)

    prediction = model.predict(
        X_processed
    )

    assert len(prediction) == 1

    assert prediction[0] in [0, 1]


def test_model_probability():
    """Check that the model produces a valid probability."""

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].iloc[[0]].copy()

    X_processed = preprocessor.transform(X)

    probability = model.predict_proba(
        X_processed
    )[0][1]

    assert 0.0 <= probability <= 1.0