import pandas as pd
from pathlib import Path


# Required columns according to the project requirements
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

# Expected categorical values
EXPECTED_LOAN_STATUS = {"Approved", "Rejected"}
EXPECTED_CREDIT_HISTORY = {0, 1}


def validate_dataset(file_path: str) -> bool:
    """
    Validate the raw loan approval dataset.

    Returns True if all critical validation checks pass.
    Raises ValueError if a critical validation check fails.
    """

    print("=" * 60)
    print("LOAN APPROVAL DATA VALIDATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Check that the file exists
    # ---------------------------------------------------------
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print("✓ Dataset file exists")

    # ---------------------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv(path)

    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---------------------------------------------------------
    # 3. Check required columns
    # ---------------------------------------------------------
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("✓ Required columns are present")

    # ---------------------------------------------------------
    # 4. Check duplicate rows
    # ---------------------------------------------------------
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate rows"
        )

    print("✓ No duplicate rows found")

    # ---------------------------------------------------------
    # 5. Check target column
    # ---------------------------------------------------------
    if df["Loan_Status"].isnull().any():
        raise ValueError("Loan_Status contains missing values")

    invalid_targets = set(df["Loan_Status"].unique()) - EXPECTED_LOAN_STATUS

    if invalid_targets:
        raise ValueError(
            f"Unexpected Loan_Status values: {invalid_targets}"
        )

    print("✓ Loan_Status is valid")

    # ---------------------------------------------------------
    # 6. Check credit history
    # ---------------------------------------------------------
    invalid_credit = set(
        df["Credit_History"].dropna().unique()
    ) - EXPECTED_CREDIT_HISTORY

    if invalid_credit:
        raise ValueError(
            f"Unexpected Credit_History values: {invalid_credit}"
        )

    print("✓ Credit_History values are valid")

    # ---------------------------------------------------------
    # 7. Check numerical columns
    # ---------------------------------------------------------
    numerical_columns = [
        "Dependents",
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Term",
        "Credit_History",
        "Age",
    ]

    for column in numerical_columns:

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"{column} should contain numeric values"
            )

    print("✓ Numerical columns have valid data types")

    # ---------------------------------------------------------
    # 8. Check for negative values
    # ---------------------------------------------------------
    non_negative_columns = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Term",
        "Age",
    ]

    for column in non_negative_columns:

        if (df[column].dropna() < 0).any():
            raise ValueError(
                f"{column} contains negative values"
            )

    print("✓ Numerical values are non-negative")

    # ---------------------------------------------------------
    # 9. Report missing values
    # ---------------------------------------------------------
    missing_values = df.isnull().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    if len(missing_values) > 0:

        print("\nMissing values detected:")
        print(missing_values)

        print(
            "\nℹ Missing values will be handled "
            "during feature engineering."
        )

    else:
        print("✓ No missing values found")

    # ---------------------------------------------------------
    # 10. Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":

    DATA_PATH = "data/raw/loan_data.csv"

    validate_dataset(DATA_PATH)