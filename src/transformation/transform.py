import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DATA_PATH = Path("data/raw/loan_data.csv")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_data.csv"

PREPROCESSOR_DIR = Path("models")
PREPROCESSOR_PATH = PREPROCESSOR_DIR / "preprocessor.joblib"


# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------

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

TARGET_COLUMN = "Loan_Status"


# ---------------------------------------------------------
# Create preprocessing pipeline
# ---------------------------------------------------------

def create_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# Transform dataset
# ---------------------------------------------------------

def transform_data():

    print("=" * 60)
    print("FEATURE ENGINEERING AND TRANSFORMATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load raw dataset
    # -----------------------------------------------------

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"✓ Loaded dataset: {df.shape}")

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    y = df[TARGET_COLUMN].map(
        {
            "Approved": 1,
            "Rejected": 0,
        }
    )

    # -----------------------------------------------------
    # Create preprocessor
    # -----------------------------------------------------

    preprocessor = create_preprocessor()

    # -----------------------------------------------------
    # Fit and transform
    # -----------------------------------------------------

    X_transformed = preprocessor.fit_transform(X)

    print(
        f"✓ Transformation completed: "
        f"{X_transformed.shape}"
    )

    # -----------------------------------------------------
    # Get feature names
    # -----------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
    )

    # Add target
    transformed_df[TARGET_COLUMN] = y.values

    # -----------------------------------------------------
    # Save processed data
    # -----------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    transformed_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Save preprocessor
    # -----------------------------------------------------

    PREPROCESSOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print(
        f"✓ Processed dataset saved to: "
        f"{PROCESSED_DATA_PATH}"
    )

    print(
        f"✓ Preprocessor saved to: "
        f"{PREPROCESSOR_PATH}"
    )

    print(
        f"✓ Final feature count: "
        f"{len(feature_names)}"
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    transform_data()