"""
Tests for the data validation stage.
"""

import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/loan_data.csv")

REQUIRED_COLUMNS = [
    "Loan_ID",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Employment_Status",
    "Applicant_Income",
    "Coapplicant_Income",
    "Loan_Amount",
    "Loan_Term",
    "Credit_History",
    "Property_Area",
    "Age",
    "Loan_Status",
]


def load_data():
    return pd.read_csv(DATA_PATH)


def test_dataset_exists():
    """Check that the raw dataset exists."""

    assert DATA_PATH.exists(), (
        f"Dataset not found at {DATA_PATH}"
    )


def test_required_columns_exist():
    """Check that all required columns are present."""

    df = load_data()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )


def test_no_duplicate_rows():
    """Check that the dataset has no duplicate rows."""

    df = load_data()

    assert df.duplicated().sum() == 0


def test_target_values_are_valid():
    """Check that Loan_Status contains only valid values."""

    df = load_data()

    valid_values = {"Approved", "Rejected"}

    actual_values = set(
        df["Loan_Status"].dropna().unique()
    )

    assert actual_values.issubset(valid_values)


def test_credit_history_values_are_valid():
    """Check that Credit_History contains only 0 or 1."""

    df = load_data()

    actual_values = set(
        df["Credit_History"].dropna().unique()
    )

    assert actual_values.issubset({0, 1})


def test_numeric_columns_are_numeric():
    """Check important numerical columns."""

    df = load_data()

    numeric_columns = [
        "Dependents",
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Term",
        "Credit_History",
        "Age",
    ]

    for column in numeric_columns:

        assert pd.api.types.is_numeric_dtype(
            df[column]
        ), f"{column} is not numeric"


def test_numeric_values_are_non_negative():
    """Check that numerical financial/demographic values are non-negative."""

    df = load_data()

    numeric_columns = [
        "Dependents",
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Term",
        "Age",
    ]

    for column in numeric_columns:

        assert (
            df[column].dropna() >= 0
        ).all(), f"{column} contains negative values"